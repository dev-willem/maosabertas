//! O que o modelo de linguagem recebe e o formato que ele devolve.
//!
//! O texto so explica o protocolo como `spec/README.md` o define. Nao ha regra
//! gramatical de Libras aqui, nem de reordenacao: isso e da equipe linguistica (regra 3).

use maosabertas_protocol::Utterance;
use serde_json::{json, Value};

/// Instrucoes de sistema. Mudar este texto muda as traducoes: registre o motivo no commit.
pub const SYSTEM: &str = "\
Você traduz enunciados em Libras para o português do Brasil. Você não vê o vídeo: \
recebe só um documento JSON no protocolo Mãos Abertas (spec 0.2), e ele é tudo o que \
se sabe sobre o enunciado.

Como ler o documento:
- `manual`: sinais em ordem de início. `gloss` é a glosa, uma notação do sinal em \
maiúsculas; glosa não é tradução, e a ordem dos sinais em Libras não segue a do \
português. Soletração vem em minúsculas com hífen (`kind: fingerspelling`); `IX(...)` é \
apontação (`kind: pointing`); `CL-*` é classificador (`kind: classifier`), cujo sentido \
depende da forma, da posição e do movimento das mãos.
- `conf` é a confiança do reconhecimento, de 0 a 1; `alt` traz outras glosas possíveis, \
da mais para a menos provável.
- `nonmanual`: expressões não manuais com tempo próprio (`t`), que podem cobrir vários \
sinais. `gramatical`: `interrogativa-sn` é pergunta de sim ou não, `interrogativa-qu` é \
pergunta aberta, `neutra` é declarativa e `duvida` depende do contexto. `cabeca`: \
`negacao` nega e `afirmacao` enfatiza. `intensidade`: grau (`baixa`, `media`, `alta`). \
`afetiva`: tom (`alegria`, `tristeza`, `susto`) da expressão sinalizada, não o estado \
emocional da pessoa.
- `fonologia`: medidas de cada mão (`esquerda`, `direita` de quem sinaliza) no \
referencial do corpo: origem no meio dos ombros, unidade = largura dos ombros, x para a \
direita de quem sinaliza, y para cima, z para a frente. `pa.pos` é onde a mão ficou; \
`o.palma` é para onde a palma olha e `o.dedos`, para onde os dedos apontam; \
`m.desloc` é para onde e quanto a mão andou e `m.trajeto`, o comprimento do caminho. \
`cat`, quando existir, é a categoria linguística. Essas medidas pesam sobretudo nos \
classificadores, para descrever posição e trajetória.
- `quality`: condições da captura. `frames_ok` baixo ou `face_occluded` alto pedem \
confiança menor.

Como responder:
- `pt`: a tradução em português natural, fiel ao que o documento sustenta. Não \
acrescente informação que ele não tenha.
- `conf`: de 0 a 1, sua confiança na tradução, considerando a confiança dos itens e a \
qualidade da captura.
- `ambiguidade`: cada ponto em que mais de uma leitura é plausível, em uma frase curta \
cada; lista vazia se não houver.
- `alternativa`: a segunda leitura mais provável, ou null se não houver.";

/// Formato da resposta. Sem `minimum`/`maximum`: saidas estruturadas nao aceitam;
/// os limites sao conferidos ao ler a resposta.
pub fn output_schema() -> Value {
    json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["pt", "conf", "ambiguidade", "alternativa"],
        "properties": {
            "pt": { "type": "string" },
            "conf": { "type": "number" },
            "ambiguidade": { "type": "array", "items": { "type": "string" } },
            "alternativa": { "anyOf": [{ "type": "string" }, { "type": "null" }] }
        }
    })
}

/// O documento que o modelo le: o enunciado inteiro, sem uma traducao anterior.
pub fn input(u: &Utterance) -> Result<String, serde_json::Error> {
    let mut u = u.clone();
    u.translation = None;
    serde_json::to_string(&u)
}
