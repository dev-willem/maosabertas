//! Traduz enunciados do protocolo: um JSON por linha na entrada, o mesmo com
//! `translation` na saida. Le so o protocolo; nunca ve video nem landmarks.
//!
//! Uso (PowerShell):
//!     $env:MAOSABERTAS_TRADUTOR = "local"          # ou "claude"
//!     $env:MAOSABERTAS_TRADUTOR_MODELO = "..."     # obrigatoria no local
//!     cargo run -p maosabertas-onnx --example transcrever -- ... |
//!         cargo run -p maosabertas-translate --example traduzir
//!
//! Variaveis em `Config::from_env`. No backend `claude` o enunciado sai da maquina:
//! so use com dado que pode sair (ADR-011, ADR-012).
//!
//! Enunciado que nao traduz sai sem `translation`, com o erro na saida de erro, e o
//! programa termina com falha.

use maosabertas_protocol::Utterance;
use maosabertas_translate::{Llm, Translator};
use std::io::{BufRead, Write};
use std::process::ExitCode;

fn main() -> ExitCode {
    let tradutor = match Llm::from_env() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("erro: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (entrada, mut saida) = (std::io::stdin().lock(), std::io::stdout().lock());
    let mut falhas = 0;
    for (n, linha) in entrada.lines().enumerate() {
        let resultado = linha.map_err(|e| e.to_string()).and_then(|l| {
            if l.trim().is_empty() {
                return Ok(None);
            }
            let mut u: Utterance = serde_json::from_str(&l).map_err(|e| e.to_string())?;
            match tradutor.translate(&u) {
                Ok(t) => u.translation = Some(t),
                Err(e) => {
                    falhas += 1;
                    eprintln!("linha {}: {}: {e}", n + 1, u.utterance_id);
                }
            }
            serde_json::to_string(&u)
                .map(Some)
                .map_err(|e| e.to_string())
        });
        match resultado {
            Ok(Some(json)) => {
                if writeln!(saida, "{json}").is_err() {
                    return ExitCode::FAILURE;
                }
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("linha {}: {e}", n + 1);
                return ExitCode::FAILURE;
            }
        }
    }
    if falhas > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
