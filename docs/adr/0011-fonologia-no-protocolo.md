# ADR-011: Fonologia no protocolo (spec v0.2)

- **Status:** aceita
- **Data:** 2026-11-20

## Contexto

O ADR-010 pôs a medição de CM, PA, O e M no núcleo. A tradução só lê o
protocolo, então o que não estiver nele se perde. A v0.1 tinha `param` com
chaves livres, sem nada definido, e nenhum item saía com parâmetro manual. O
ADR-009 (seção 5) listou o que o schema precisava decidir: chaves de `param`,
confiança por parâmetro, mão do item e itens sobrepostos.

Há dois limites que o schema não pode cruzar:

- **Regra 3:** nome de configuração, locação, orientação ou movimento é decisão
  linguística, e os inventários ainda não existem.
- **Regra 1 / LGPD:** landmarks de mão e rosto são dado biométrico, e o protocolo
  sai da máquina (tradução por API, ADR-012; registros; retreino).

## Decisão

1. **Dois níveis por parâmetro.** A *medida*, geométrica e calculada pelo núcleo,
   sai já. A *categoria* (`cat` e `conf`) só aceita valor de inventário publicado
   pela equipe linguística. Até lá fica ausente, e `conf` sem `cat` é inválido no
   schema. Assim o caminho até a tradução funciona sem que ninguém invente
   categoria.
2. **Por mão.** `fonologia` é uma lista com uma entrada por mão que articula o
   item (`mao`: `esquerda` ou `direita`, de quem sinaliza). Itens podem se
   sobrepor no tempo, o que cobre classificadores simultâneos (figura e fundo). Não
   há campo de mão no item: a lista já diz quais mãos articulam.
3. **`param` sai.** Era livre e não tinha produtor; os parâmetros vivem em
   `fonologia`. É mudança incompatível, então a versão sobe para 0.2.
4. **O que sai no protocolo** (resposta à pergunta 4 do plano: valores contínuos
   sim, agregados):

   | Sai | Não sai |
   | --- | --- |
   | Agregados por item: centro da palma, normal da palma, direção dos dedos, deslocamento, comprimento do caminho, fração visível | Landmarks, trajetória quadro a quadro, descritor de forma da mão (geometria da mão é biometria conhecida) |
   | Duas casas decimais (0,01 largura de ombro) | Precisão maior, que não ajuda a tradução e facilita a reidentificação |

5. **Profundidade reservada, não emitida.** `pa.prof` existe no schema, mas o
   núcleo não emite. O z do punho na pose tem cerca de 8 vezes o ruído de x e y e
   um deslocamento de 0,30 largura de ombro entre sinalizantes (experimento 006).
   Ele passa a ser emitido quando houver validação contra referência de
   profundidade real.
6. **Como o núcleo agrega** (`phonology::measure`, chamado pelo `Transcriber`):
   - Trecho do item: o pico do CTC estendido até o meio da folga com cada vizinho,
     sem passar de 0,25 s de cada lado, porque o intervalo do CTC é só o pico. O
     limite impede que a pausa que fecha o enunciado (a mão descendo ao repouso)
     entre inteira e derrube a fração visível de um sinal isolado (achado da
     revisão). Isso é um `shortcut`, a trocar por alinhamento.
   - A mão entra se ficar no espaço de sinalização em pelo menos metade dos
     quadros do trecho (`shortcut`, sem calibração).
   - PA e O: mediana nos quadros de segurar; sem segurar, em todos os visíveis.
   - M: centro da palma do primeiro ao último quadro visível e o caminho entre eles.

## Verificação

- O enunciado de referência dos testes (`crates/maosabertas-core/tests/data`) é
  gerado pelo núcleo com mãos sintéticas e validado contra o schema no CI.
- De ponta a ponta com dado real (fora do repositório): modelo base, clipe de um
  sinalizante de teste, `examples/transcrever.rs`. A glosa sai certa, com a
  fonologia das duas mãos, e o JSON valida contra o schema. A mão esquerda
  cruzando para o lado direito foi conferida contra os punhos da pose: o eixo
  está certo.

## Alternativas consideradas

| Alternativa | Por que não |
| --- | --- |
| Manter `param` livre e preencher com chaves do núcleo | Duas formas de dizer a mesma coisa; nenhum consumidor saberia o que esperar |
| Emitir grupos sem nome do descritor de CM ("grupo 7") | O ADR-009 já veta: grupo sem nome não é categoria, e a tradução trataria como se fosse |
| Mandar a trajetória reamostrada (45 pontos) | Biométrica e grande; a tradução precisa de "para onde e quanto", não da curva |
| Campo `hand` no item | Redundante com a lista por mão |
| Profundidade já na v0.2 | Medida ruidosa e enviesada (item 5) |

## Pedido à equipe linguística

Vai como issue com o modelo "Questão linguística":

1. Inventário de configurações de mão do projeto (vira `cm.cat`).
2. Regiões de ponto de articulação: se e como recortar o espaço contínuo de
   `pa.pos` em categorias.
3. Categorias de orientação e de movimento, e se tipos de movimento de bases
   públicas servem de ponto de partida.
4. Inventário e nomes dos classificadores (`CL-*`) e o critério de fronteira entre
   sinal lexical e uso classificador.
5. Construções com as duas mãos e como anotá-las.

## Consequências

- A v0.2 quebra a v0.1: consumidores precisam trocar `param` por `fonologia`.
- `Transcriber::push` recebe as mãos de cada quadro (lista vazia = sem medição).
- A tradução já recebe locação, orientação e movimento medidos, mas ainda sem
  categoria. O quanto isso ajuda a traduzir não foi medido (regra 4).
