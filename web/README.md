# Casca web

TypeScript, o mínimo possível: `getUserMedia`, canvas e DOM. A lógica
está no núcleo em Rust compilado para WASM.

Vazia por enquanto. Depende de aprovar as dependências (TypeScript, Vite,
MediaPipe Tasks, wasm-bindgen) e de decidir como a inferência roda no navegador
(ADR-005). Hoje o teste com câmera é a ferramenta Python (`docs/GUIA.md`, seção 3).
