# ADR-001: Núcleo em Rust, Python só no treino

- **Status:** aceita
- **Data:** 2026-10-14

## Contexto

O caminho de tempo real precisa caber em ~33 ms por quadro, rodar no
dispositivo de quem sinaliza (o codec da chamada destrói o detalhe das
mãos) e ser integrável em qualquer sistema. Quanto mais linguagens, mais
reimplementações do mesmo protocolo e mais divergência.

## Decisão

1. **Rust** em todo o caminho de tempo real: features, inferência,
   decodificação CTC, escopo temporal, servidor e bindings. Um crate
   compila para nativo e para `wasm32-unknown-unknown`.
2. **Python** apenas para treinar, avaliar e exportar ONNX. Não entra em
   produção.
3. **TypeScript** apenas como casca: câmera, canvas e DOM. Nenhuma lógica.
4. PHP, NestJS e Java ficam de fora.

## Alternativas consideradas

| Alternativa | Por que não |
| --- | --- |
| Python em produção | Interpretador e GIL no caminho crítico |
| NestJS para a API | Mais um runtime sem ganho sobre `axum` |
| Java | Ecossistema de ML pobre; sem vantagem aqui |

## Consequências

- O mesmo código roda no cliente e no servidor; só muda o alvo.
- Curva de aprendizado maior no início.
- A extração de pose depende do MediaPipe: os modelos serão convertidos
  para ONNX e rodados via crate `ort`; na fase inicial, MediaPipe Tasks
  via WASM na casca. Isso ainda precisa ser validado na prática.
