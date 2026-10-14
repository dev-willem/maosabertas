//! Tipos do protocolo de transcricao de Libras.
//!
//! A fonte da verdade e `spec/utterance.schema.json`. Estes tipos devem
//! acompanhar o schema; ao mudar um, mude o outro e suba a versao da
//! especificacao.
//!
//! Glosa e notacao, nao traducao. A ordem dos sinais em Libras nao e fixa:
//! nada aqui deve reordenar nada.

use serde::{Deserialize, Serialize};

/// Versao da especificacao que este crate implementa.
pub const SPEC_VERSION: &str = "0.1";

/// Intervalo `[inicio, fim]` em segundos.
pub type Span = [f64; 2];

/// Um enunciado: trilhas manuais e nao manuais alinhadas no tempo.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Utterance {
    pub spec_version: String,
    pub utterance_id: String,
    pub t_start: f64,
    pub t_end: f64,
    pub manual: Vec<ManualItem>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nonmanual: Vec<NonManualItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality: Option<Quality>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub translation: Option<Translation>,
}

/// Um sinal, classificador, soletracao ou apontacao.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ManualItem {
    pub gloss: String,
    #[serde(default)]
    pub kind: ManualKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<serde_json::Value>,
    pub t: Span,
    pub conf: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub alt: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ManualKind {
    #[default]
    Lexical,
    Classifier,
    Fingerspelling,
    Pointing,
}

/// Marcador nao manual. Tem escopo proprio: nunca pertence a um sinal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NonManualItem {
    pub tier: Tier,
    pub label: NonManualLabel,
    pub t: Span,
    pub conf: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    Gramatical,
    Afetiva,
    Intensidade,
    Cabeca,
    Olhar,
}

/// Conjunto fechado na v0.1. Mudar exige nova versao da especificacao.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NonManualLabel {
    Neutra,
    InterrogativaSn,
    InterrogativaQu,
    Negacao,
    Afirmacao,
    Duvida,
    Alegria,
    Tristeza,
    Susto,
    Baixa,
    Media,
    Alta,
}

/// Condicoes da captura, para filtrar amostras e avisar o usuario.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Quality {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frames_ok: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub face_occluded: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Translation {
    pub pt: String,
    pub conf: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ambiguidade: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alternativa: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl Utterance {
    /// Marcadores cujo escopo cobre o instante `t`.
    pub fn markers_at(&self, t: f64) -> impl Iterator<Item = &NonManualItem> {
        self.nonmanual
            .iter()
            .filter(move |m| m.t[0] <= t && t <= m.t[1])
    }

    /// Linha de glosa para exibicao. Nao e traducao.
    pub fn gloss_line(&self) -> String {
        self.manual
            .iter()
            .map(|m| m.gloss.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXEMPLO: &str = include_str!("../../../spec/examples/01-declarativa.json");

    #[test]
    fn le_o_exemplo_da_spec() {
        let u: Utterance = serde_json::from_str(EXEMPLO).expect("exemplo valido");
        assert_eq!(u.spec_version, SPEC_VERSION);
        assert_eq!(u.gloss_line(), "QUEIJO RATO COMER");
    }

    #[test]
    fn marcador_cobre_varios_sinais() {
        let u: Utterance = serde_json::from_str(EXEMPLO).unwrap();
        assert_eq!(u.markers_at(0.3).count(), 1);
        assert_eq!(u.markers_at(1.5).count(), 1);
        assert_eq!(u.markers_at(99.0).count(), 0);
    }

    #[test]
    fn ida_e_volta_preserva_o_documento() {
        let u: Utterance = serde_json::from_str(EXEMPLO).unwrap();
        let texto = serde_json::to_string(&u).unwrap();
        let v: Utterance = serde_json::from_str(&texto).unwrap();
        assert_eq!(u, v);
    }
}
