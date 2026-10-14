# ADR-007: Fim de enunciado por pausa manual ou duração máxima

- **Status:** aceita
- **Data:** 2026-10-14

## Contexto

O núcleo recebe probabilidades por quadro sem fim, mas a unidade do schema
é o enunciado. Alguém precisa decidir onde um enunciado acaba para emitir o
`Utterance` completo. Ainda não há modelo treinado nem dados anotados para
calibrar nem para medir qualquer regra.

## Decisão

1. O `Transcriber` (`crates/maosabertas-core/src/transcriber.rs`) fecha o
   enunciado quando o branco vence no modelo manual em quadros seguidos por
   pelo menos `pause` segundos, ou quando o enunciado chega a
   `max_duration` segundos. Os dois valores são obrigatórios e sem padrão:
   dependem do modelo e precisam de calibração com dados.
2. Os enunciados cobrem o fluxo sem buracos: cada um vai do corte anterior
   ao seu, com a pausa dentro. `t_start`/`t_end` são esse intervalo.
3. Enunciado sem nenhum item, manual ou não manual, não sai e não gasta
   número em `utterance_id` (prefixo do chamador + contador a partir de 1).
4. Marcador não manual aberto no corte fecha ali e reabre no enunciado
   seguinte, sem passar de novo pelo limiar de entrada. O marcador tem
   escopo próprio, mas o enunciado é a unidade do schema.
5. Sinal em curso num corte por duração sai partido nos dois enunciados.
6. `quality` é calculado por enunciado: `frames_ok` é a fração de quadros
   com referência (ombros), `face_occluded` a fração sem rosto e `fps` o
   configurado. Quem extrai os landmarks informa, por quadro, o que viu.

## Alternativas consideradas

| Alternativa | Por que não |
| --- | --- |
| Janela fixa de N segundos | Corta no meio de sinais sempre, não só em enunciado longo demais |
| Usar marcadores não manuais para decidir o fim | Seria regra gramatical de Libras; só com linguistas e pessoas surdas |
| Modelo treinado de fronteira de enunciado | Precisa de corpus anotado que ainda não existe |
| Cortar no início da pausa | Exige reprocessar quadros já entregues ou atrasar a saída em `pause` |
| Marcador aberto no corte fica só no primeiro enunciado | Perde o resto do escopo quando a probabilidade fica entre saída e entrada |

## Consequências

- A pausa é heurística de engenharia, não afirmação sobre a língua. Pausa
  curta demais parte frases com hesitação; longa demais junta frases.
  Nenhum dos dois efeitos foi medido.
- `t_end` inclui a pausa; quem precisa do fim do último sinal lê `manual`.
- Silêncio longo com marcador ativo vira vários enunciados só não manuais,
  um a cada `pause` segundos.
- `min_frames` da histerese vale por trecho: a ponta curta de um marcador
  cortado pode se perder.
- `fps` em `quality` é o configurado; medir o real exige carimbo de tempo
  por quadro, que o transcritor ainda não recebe.
