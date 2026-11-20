//! Camada de traducao (ADR-010 e ADR-012): recebe um [`Utterance`] e nada mais,
//! nem video nem landmarks, e devolve a [`Translation`] em portugues.
//!
//! Dois backends com a mesma interface, trocados por configuracao:
//! - `claude`: API de mensagens da Anthropic;
//! - `local`: servidor compativel com a API de chat da OpenAI rodando na maquina
//!   (Ollama, llama.cpp, vLLM). Nada sai da maquina.
//!
//! Configuracao por variaveis de ambiente (ver [`Config::from_env`]).

mod prompt;

pub use prompt::{output_schema, SYSTEM};

use maosabertas_protocol::{Translation, Utterance};
use serde::Deserialize;
use serde_json::{json, Value};
use std::fmt;
use std::time::Duration;

/// Quem traduz um enunciado fechado.
pub trait Translator {
    fn translate(&self, u: &Utterance) -> Result<Translation, Error>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Claude,
    Local,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Backend::Claude => "claude",
            Backend::Local => "local",
        })
    }
}

/// Modelo padrao do backend `claude`.
pub const CLAUDE_MODEL: &str = "claude-opus-5-5";
const CLAUDE_URL: &str = "https://api.anthropic.com";
const LOCAL_URL: &str = "http://localhost:11434";
const ANTHROPIC_VERSION: &str = "2026-10-21";
/// Recusa por politica e refeita em outro modelo dentro da mesma chamada.
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
/// Raciocinio do Claude: padrao da API para o Opus 5.5, explicito para nao mudar sozinho.
const EFFORT: &str = "medium";
const MAX_TOKENS: u32 = 16_000;
const TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub backend: Backend,
    pub model: String,
    /// Raiz do servidor, sem `/v1/...`.
    pub url: String,
    /// Chave da API: obrigatoria no `claude`, opcional no `local`.
    pub key: Option<String>,
}

impl Config {
    /// Le do ambiente:
    ///
    /// | Variavel | `claude` | `local` |
    /// | --- | --- | --- |
    /// | `MAOSABERTAS_TRADUTOR` | `claude` | `local` |
    /// | `MAOSABERTAS_TRADUTOR_MODELO` | padrao [`CLAUDE_MODEL`] | obrigatoria |
    /// | `MAOSABERTAS_TRADUTOR_URL` | padrao `https://api.anthropic.com` | padrao `http://localhost:11434` |
    /// | chave | `ANTHROPIC_API_KEY`, obrigatoria | `MAOSABERTAS_TRADUTOR_CHAVE`, opcional |
    pub fn from_env() -> Result<Self, Error> {
        Self::from_lookup(|k| std::env::var(k).ok())
    }

    /// Como [`from_env`](Self::from_env), lendo de `get` (testes).
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<Self, Error> {
        let var = |k: &str| {
            get(k)
                .map(|v| v.trim().to_owned())
                .filter(|v| !v.is_empty())
        };
        let (backend, default_url, key) = match var("MAOSABERTAS_TRADUTOR").as_deref() {
            Some("claude") => (Backend::Claude, CLAUDE_URL, var("ANTHROPIC_API_KEY")),
            Some("local") => (Backend::Local, LOCAL_URL, var("MAOSABERTAS_TRADUTOR_CHAVE")),
            other => {
                return Err(Error::Config(format!(
                    "MAOSABERTAS_TRADUTOR precisa ser `claude` ou `local`, recebido {other:?}"
                )))
            }
        };
        let model = match (var("MAOSABERTAS_TRADUTOR_MODELO"), backend) {
            (Some(m), _) => m,
            (None, Backend::Claude) => CLAUDE_MODEL.to_owned(),
            (None, Backend::Local) => {
                return Err(Error::Config(
                    "MAOSABERTAS_TRADUTOR_MODELO e obrigatoria no backend local".into(),
                ))
            }
        };
        if backend == Backend::Claude && key.is_none() {
            return Err(Error::Config(
                "ANTHROPIC_API_KEY e obrigatoria no backend claude".into(),
            ));
        }
        let url = var("MAOSABERTAS_TRADUTOR_URL").unwrap_or_else(|| default_url.to_owned());
        Ok(Self {
            backend,
            model,
            url: url.trim_end_matches('/').to_owned(),
            key,
        })
    }
}

/// Envio de um POST com corpo JSON. Devolve status e corpo da resposta.
pub trait Http {
    fn post(&self, url: &str, headers: &[(&str, &str)], body: &str)
        -> Result<(u16, String), Error>;
}

/// [`Http`] real, com timeout e TLS nativo. Status de erro volta como resposta, para
/// a mensagem do servidor chegar ao erro.
pub struct Ureq(ureq::Agent);

impl Default for Ureq {
    fn default() -> Self {
        let tls = ureq::tls::TlsConfig::builder()
            .provider(ureq::tls::TlsProvider::NativeTls)
            .build();
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(TIMEOUT))
            .tls_config(tls)
            .build();
        Self(agent.into())
    }
}

impl Http for Ureq {
    fn post(
        &self,
        url: &str,
        headers: &[(&str, &str)],
        body: &str,
    ) -> Result<(u16, String), Error> {
        let net = |e: ureq::Error| Error::Network(e.to_string());
        let mut req = self.0.post(url).content_type("application/json");
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let mut resp = req.send(body).map_err(net)?;
        let status = resp.status().as_u16();
        let text = resp.body_mut().read_to_string().map_err(net)?;
        Ok((status, text))
    }
}

/// Tradutor por modelo de linguagem, num dos dois backends.
pub struct Llm<H = Ureq> {
    config: Config,
    http: H,
}

impl Llm {
    pub fn new(config: Config) -> Self {
        Self::with_http(config, Ureq::default())
    }

    /// Le a configuracao do ambiente ([`Config::from_env`]).
    pub fn from_env() -> Result<Self, Error> {
        Ok(Self::new(Config::from_env()?))
    }
}

impl<H: Http> Llm<H> {
    pub fn with_http(config: Config, http: H) -> Self {
        Self { config, http }
    }

    fn claude(&self, input: String) -> Result<String, Error> {
        let c = &self.config;
        let body = json!({
            "model": c.model,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM,
            "messages": [{ "role": "user", "content": input }],
            "output_config": {
                "effort": EFFORT,
                "format": { "type": "json_schema", "schema": output_schema() }
            },
            "fallbacks": "default"
        });
        let key = c.key.as_deref().unwrap_or_default();
        let headers = [
            ("x-api-key", key),
            ("anthropic-version", ANTHROPIC_VERSION),
            ("anthropic-beta", FALLBACK_BETA),
        ];
        let v = self.send(&format!("{}/v1/messages", c.url), &headers, &body)?;
        match v["stop_reason"].as_str() {
            Some("refusal") => return Err(Error::Refusal),
            Some("max_tokens") => return Err(Error::Truncated),
            _ => {}
        }
        v["content"]
            .as_array()
            .and_then(|blocks| blocks.iter().find(|b| b["type"] == "text"))
            .and_then(|b| b["text"].as_str())
            .map(str::to_owned)
            .ok_or_else(|| Error::Response("resposta sem bloco de texto".into()))
    }

    fn local(&self, input: String) -> Result<String, Error> {
        let c = &self.config;
        let body = json!({
            "model": c.model,
            "messages": [
                { "role": "system", "content": SYSTEM },
                { "role": "user", "content": input }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": { "name": "traducao", "strict": true, "schema": output_schema() }
            }
        });
        let auth = c.key.as_deref().map(|k| format!("Bearer {k}"));
        let headers: Vec<(&str, &str)> = auth
            .as_deref()
            .map(|a| ("authorization", a))
            .into_iter()
            .collect();
        let v = self.send(&format!("{}/v1/chat/completions", c.url), &headers, &body)?;
        let choice = &v["choices"][0];
        if choice["finish_reason"] == "length" {
            return Err(Error::Truncated);
        }
        choice["message"]["content"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| Error::Response("resposta sem choices[0].message.content".into()))
    }

    fn send(&self, url: &str, headers: &[(&str, &str)], body: &Value) -> Result<Value, Error> {
        let (status, text) = self.http.post(url, headers, &body.to_string())?;
        if !(200..300).contains(&status) {
            return Err(Error::Http {
                status,
                body: text.chars().take(500).collect(),
            });
        }
        serde_json::from_str(&text).map_err(|e| Error::Response(format!("JSON invalido: {e}")))
    }
}

impl<H: Http> Translator for Llm<H> {
    fn translate(&self, u: &Utterance) -> Result<Translation, Error> {
        let input = prompt::input(u).map_err(|e| Error::Response(e.to_string()))?;
        let text = match self.config.backend {
            Backend::Claude => self.claude(input)?,
            Backend::Local => self.local(input)?,
        };
        let model = format!("{}:{}", self.config.backend, self.config.model);
        parse(&text, model)
    }
}

#[derive(Deserialize)]
struct Output {
    pt: String,
    conf: f64,
    #[serde(default)]
    ambiguidade: Vec<String>,
    #[serde(default)]
    alternativa: Option<String>,
}

/// Le e confere a resposta do modelo: o formato pedido nao garante os limites.
fn parse(text: &str, model: String) -> Result<Translation, Error> {
    let out: Output = serde_json::from_str(text)
        .map_err(|e| Error::Response(format!("fora do formato pedido: {e}")))?;
    if out.pt.trim().is_empty() {
        return Err(Error::Response("traducao vazia".into()));
    }
    if !(0.0..=1.0).contains(&out.conf) {
        return Err(Error::Response(format!(
            "conf fora de [0, 1]: {}",
            out.conf
        )));
    }
    Ok(Translation {
        pt: out.pt.trim().to_owned(),
        conf: out.conf,
        ambiguidade: out.ambiguidade,
        alternativa: out.alternativa.filter(|a| !a.trim().is_empty()),
        model: Some(model),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Config(String),
    /// Falha de rede ou de TLS.
    Network(String),
    /// Status fora de 2xx; `body` com no maximo 500 caracteres.
    Http {
        status: u16,
        body: String,
    },
    /// O modelo recusou (e o modelo de reserva tambem).
    Refusal,
    /// A resposta acabou no limite de tokens.
    Truncated,
    /// Resposta fora do formato combinado.
    Response(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Config(m) => write!(f, "configuracao do tradutor: {m}"),
            Error::Network(m) => write!(f, "rede: {m}"),
            Error::Http { status, body } => write!(f, "HTTP {status}: {body}"),
            Error::Refusal => f.write_str("o modelo recusou traduzir"),
            Error::Truncated => f.write_str("resposta cortada no limite de tokens"),
            Error::Response(m) => write!(f, "resposta invalida: {m}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    const ENUNCIADO: &str = include_str!("../../../spec/examples/03-classificador.json");

    /// URL, cabecalhos e corpo de uma requisicao.
    type Pedido = (String, Vec<(String, String)>, Value);

    /// Grava a requisicao e devolve uma resposta pronta.
    struct Falso {
        resposta: (u16, String),
        pedido: RefCell<Option<Pedido>>,
    }

    impl Falso {
        fn new(status: u16, body: Value) -> Self {
            Self {
                resposta: (status, body.to_string()),
                pedido: RefCell::new(None),
            }
        }
    }

    impl Http for &Falso {
        fn post(
            &self,
            url: &str,
            headers: &[(&str, &str)],
            body: &str,
        ) -> Result<(u16, String), Error> {
            let headers = headers
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            let body = serde_json::from_str(body).expect("corpo JSON");
            *self.pedido.borrow_mut() = Some((url.into(), headers, body));
            Ok(self.resposta.clone())
        }
    }

    fn config(backend: Backend) -> Config {
        Config {
            backend,
            model: "modelo-x".into(),
            url: "http://servidor".into(),
            key: Some("segredo".into()),
        }
    }

    fn enunciado() -> Utterance {
        let mut u: Utterance = serde_json::from_str(ENUNCIADO).unwrap();
        u.translation = Some(Translation {
            pt: "traducao antiga".into(),
            conf: 0.1,
            ambiguidade: vec![],
            alternativa: None,
            model: None,
        });
        u
    }

    fn saida() -> String {
        json!({
            "pt": " O carro andou e caiu. ",
            "conf": 0.6,
            "ambiguidade": ["queda ou descida"],
            "alternativa": ""
        })
        .to_string()
    }

    #[test]
    fn claude_pede_formato_e_le_o_texto() {
        let falso = Falso::new(
            200,
            json!({
                "stop_reason": "end_turn",
                "content": [
                    { "type": "thinking", "thinking": "" },
                    { "type": "text", "text": saida() }
                ]
            }),
        );
        let t = Llm::with_http(config(Backend::Claude), &falso)
            .translate(&enunciado())
            .unwrap();
        assert_eq!(t.pt, "O carro andou e caiu.");
        assert_eq!(t.conf, 0.6);
        assert_eq!(t.ambiguidade, ["queda ou descida"]);
        assert_eq!(t.alternativa, None);
        assert_eq!(t.model.as_deref(), Some("claude:modelo-x"));

        let (url, headers, body) = falso.pedido.take().unwrap();
        assert_eq!(url, "http://servidor/v1/messages");
        let h: HashMap<_, _> = headers.into_iter().collect();
        assert_eq!(h["x-api-key"], "segredo");
        assert_eq!(h["anthropic-version"], ANTHROPIC_VERSION);
        assert_eq!(h["anthropic-beta"], FALLBACK_BETA);
        assert_eq!(body["model"], "modelo-x");
        assert_eq!(body["system"], SYSTEM);
        assert_eq!(body["fallbacks"], "default");
        assert_eq!(body["output_config"]["format"]["schema"], output_schema());
        // O modelo le o protocolo, e so ele: sem a traducao anterior.
        let lido: Utterance =
            serde_json::from_str(body["messages"][0]["content"].as_str().unwrap()).unwrap();
        assert_eq!(lido.translation, None);
        assert_eq!(lido.manual, enunciado().manual);
    }

    #[test]
    fn local_usa_chat_compativel() {
        let falso = Falso::new(
            200,
            json!({ "choices": [{ "finish_reason": "stop", "message": { "content": saida() } }] }),
        );
        let t = Llm::with_http(config(Backend::Local), &falso)
            .translate(&enunciado())
            .unwrap();
        assert_eq!(t.model.as_deref(), Some("local:modelo-x"));

        let (url, headers, body) = falso.pedido.take().unwrap();
        assert_eq!(url, "http://servidor/v1/chat/completions");
        assert_eq!(headers, [("authorization".into(), "Bearer segredo".into())]);
        assert_eq!(body["messages"][0]["content"], SYSTEM);
        assert_eq!(
            body["response_format"]["json_schema"]["schema"],
            output_schema()
        );
    }

    fn erro(backend: Backend, status: u16, resposta: Value) -> Error {
        let falso = Falso::new(status, resposta);
        Llm::with_http(config(backend), &falso)
            .translate(&enunciado())
            .unwrap_err()
    }

    #[test]
    fn recusa_corte_http_e_formato_viram_erro() {
        let claude = |stop: &str| json!({ "stop_reason": stop, "content": [] });
        assert_eq!(
            erro(Backend::Claude, 200, claude("refusal")),
            Error::Refusal
        );
        assert_eq!(
            erro(Backend::Claude, 200, claude("max_tokens")),
            Error::Truncated
        );
        assert!(matches!(
            erro(Backend::Claude, 200, claude("end_turn")),
            Error::Response(_)
        ));
        assert_eq!(
            erro(Backend::Claude, 429, json!({ "error": "lento" })),
            Error::Http {
                status: 429,
                body: r#"{"error":"lento"}"#.into()
            }
        );
        let local = |content: Value| json!({ "choices": [{ "message": { "content": content } }] });
        let fora = |v: Value| erro(Backend::Local, 200, local(json!(v.to_string())));
        assert!(matches!(
            fora(json!({ "pt": "x", "conf": 1.5, "ambiguidade": [], "alternativa": null })),
            Error::Response(m) if m.contains("conf")
        ));
        assert!(matches!(
            fora(json!({ "pt": " ", "conf": 0.5, "ambiguidade": [], "alternativa": null })),
            Error::Response(_)
        ));
        assert!(matches!(
            erro(Backend::Local, 200, local(json!("nao e json"))),
            Error::Response(_)
        ));
        assert_eq!(
            erro(
                Backend::Local,
                200,
                json!({ "choices": [{ "finish_reason": "length", "message": {} }] })
            ),
            Error::Truncated
        );
    }

    fn le(vars: &[(&str, &str)]) -> Result<Config, Error> {
        let map: HashMap<String, String> = vars
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Config::from_lookup(|k| map.get(k).cloned())
    }

    #[test]
    fn configuracao_pelo_ambiente() {
        let c = le(&[
            ("MAOSABERTAS_TRADUTOR", "claude"),
            ("ANTHROPIC_API_KEY", "k"),
        ])
        .unwrap();
        assert_eq!(
            c,
            Config {
                backend: Backend::Claude,
                model: CLAUDE_MODEL.into(),
                url: CLAUDE_URL.into(),
                key: Some("k".into()),
            }
        );
        let c = le(&[
            ("MAOSABERTAS_TRADUTOR", "local"),
            ("MAOSABERTAS_TRADUTOR_MODELO", "m"),
            ("MAOSABERTAS_TRADUTOR_URL", "http://outro:8080/"),
        ])
        .unwrap();
        assert_eq!(
            (c.backend, c.url.as_str(), c.key),
            (Backend::Local, "http://outro:8080", None)
        );

        for vars in [
            &[][..],
            &[("MAOSABERTAS_TRADUTOR", "outro")],
            &[("MAOSABERTAS_TRADUTOR", "claude")],
            &[
                ("MAOSABERTAS_TRADUTOR", "claude"),
                ("ANTHROPIC_API_KEY", " "),
            ],
            &[("MAOSABERTAS_TRADUTOR", "local")],
        ] {
            assert!(matches!(le(vars), Err(Error::Config(_))), "{vars:?}");
        }
    }
}
