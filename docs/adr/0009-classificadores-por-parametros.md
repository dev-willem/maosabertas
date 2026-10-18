# ADR-009: Classificadores por parâmetros fonológicos

- **Status:** proposta. Depende das decisões linguísticas e de schema da
  seção 5; até lá, o núcleo não emite `kind: classifier`.
- **Data:** 2026-10-17

## 1. Contexto

Classificador (CL) é configuração de mão produtiva: a mesma forma de mão
representa uma classe de entidade e varia livremente em movimento,
locação e orientação para descrever trajetória, forma ou manipulação. Um
reconhecedor só de glosas fechadas (o CTC do ADR-008) não cobre isso: o
número de combinações é aberto.

### O que já existe

| Onde | O que há |
| --- | --- |
| `spec/utterance.schema.json` | `manualItem.kind` aceita `classifier`; `param` é objeto de **chaves livres** na v0.1; a glosa `CL-*` é só convenção de texto |
| `spec/README.md` | Notação `CL-VEICULO(...)` + `param`, como exemplo; rótulos ainda sem revisão linguística |
| `spec/examples/03-classificador.json` | Único exemplo: `CL-VEICULO` com `param` `{movimento, plano}` (valores ilustrativos, não validados) |
| `spec/CHANGELOG.md` | "Parâmetros de classificador: `param` ainda tem chaves livres" está em aberto |
| `maosabertas-protocol` | `ManualKind::Classifier` e `param: Option<Value>` existem |
| `maosabertas-core/ctc.rs` | Todo item sai com `param: None`; o `kind` vem do vocabulário. Nada produz classificador |
| `features.rs` | 203 features: mãos com 21 pontos normalizados pelos ombros (contêm locação e orientação implícitas), sem descritor de forma da mão |
| `training/` | Nada específico para CL |

### O que falta

Inventário de configurações de mão e de CL; chaves e valores de `param`;
dados anotados com CL; qualquer módulo que estime parâmetros
fonológicos; regra para separar sinal lexical de classificador; guia de
anotação de CL.

## 2. Proposta

Reconhecer **parâmetros**, não classificadores inteiros. Cada parâmetro é
estimado por um módulo próprio, causal sempre que possível, e o item
`CL-*` é a composição deles sobre um trecho de tempo.

```
quadro ─┬─ features (203) ──── TCN causal + CTC ──── segmentos e glosas lexicais
        ├─ forma da mão (CM) ── descritor invariante ── grupo de CM por quadro
        ├─ locação (L) ──────── mão vs. ombros/cabeça ─ posição por quadro
        ├─ orientação (Or) ──── normal da palma ─────── vetor por quadro
        └─ movimento (M) ────── trajetória do centro ── tipo por segmento fechado
```

### 2.1 Parâmetros

| Parâmetro | Como | Causal? | Onde vive |
| --- | --- | --- | --- |
| **CM** | Descritor de 34 valores dos 21 pontos: 15 flexões, 4 aberturas, 15 distâncias entre pontas e até o punho, divididas pela palma. Invariante a rotação, escala, translação e lado (protótipo medido, seção 6) | sim, lookahead 0 | núcleo (Rust), como `features.rs` (ADR-004) |
| **L** | Punho e centro da mão no referencial dos ombros e da cabeça, que as 203 features já têm | sim | núcleo |
| **Or** | Normal da palma (pontos 0, 5, 17) e direção dos dedos no referencial dos ombros | sim | núcleo |
| **M** | Trajetória do centro da mão no trecho, reamostrada (ex.: 45 pontos) | **não**: precisa do trecho fechado | núcleo, chamado quando o item fecha |

O descritor de CM, de L e de Or é geométrico e determinístico: não precisa
de modelo para existir. O que precisa de dado é **mapear** o valor contínuo
para uma categoria. Até haver inventário, a saída é "grupo N" (agrupamento
sem nome), só para pesquisa, nunca no protocolo.

### 2.2 Lexical vs. classificador

1. O modelo CTC continua dono da **segmentação**: ele decide onde há item
   manual. Glosa lexical sai como hoje.
2. Quando o vocabulário tiver CL (decisão linguística), o CTC ganha um
   token por CL (`CL-<nome definido pela equipe>`). O CTC só aprende a
   detectar "aqui há um CL desta classe"; a variação produtiva fica de fora.
3. Sobre o trecho do token CL, os módulos de parâmetro preenchem `param`.
   CM confirma ou contesta a classe; L, Or e M descrevem o uso.
4. Sinal sem token conhecido, com CM estável de um grupo associado a CL e
   M/L fora de qualquer entrada lexical, vira **candidato** a CL para
   anotação humana, não saída.

Não há regra fixa que transforme glosa em CL nem o contrário. A fronteira
entre sinal congelado e uso produtivo é decisão linguística.

### 2.3 Encaixe na arquitetura atual

- **Tempo real e lookahead 0:** CM, L e Or são por quadro. M exige o fim do
  trecho: `param.movimento` só é preenchido quando o item fecha, como o
  `Transcriber` já faz com enunciados (ADR-007). O atraso é o do próprio
  trecho; precisa ser medido.
- **Contrato ONNX (ADR-005):** inalterado para o modelo manual. Se um
  parâmetro precisar de modelo aprendido, ele é **outro arquivo ONNX com o
  mesmo contrato** `[T, FEATURE_DIM] → [T, C_p]`, softmax sobre as
  categorias daquele parâmetro. Nada de várias cabeças num arquivo só.
- **ADR-004:** o descritor tem de ir para `maosabertas-core` e chegar ao
  treino via `maosabertas-py`. O protótipo em Python é descartável.
  Acrescentá-lo às 203 features muda `FEATURE_DIM` e invalida modelos:
  decidir com medição, não agora.

## 3. De onde vêm os dados

As fontes concretas e os termos de uso de cada uma ficam no registro de
experimentos, fora do repositório.

| Tipo de fonte | Serve para | Limite |
| --- | --- | --- |
| **Base pública de sinais isolados**, vários sinalizantes | Validar que o descritor de CM é estável **entre pessoas** (feito, seção 6) | Sinais lexicais, sem CL; o rótulo é o sinal, não a CM |
| **Vídeos agrupados por configuração de mão** (com autorização) | **Supervisão fraca de CM:** clipes de um mesmo grupo formam um positivo **sem nome**. Serve para aprender métrica (mesmo grupo perto, grupos diferentes longe) ou para validar o descritor | Um sinalizante só não mede generalização entre pessoas; a CM pode durar só parte do clipe (usar quadros de segurar) |
| **Base pública de trajetórias de movimento** | Testar a derivação da trajetória e um primeiro reconhecedor de M (feito, seção 6) | Classes da base, não do projeto; há inconsistência entre nome e amplitude de algumas classes (seção 6.2) |
| **Corpus anotado com CL** | Única fonte para detectar CL e avaliar `param` | Não existe no projeto. Exige guia de anotação, linguistas, pessoas surdas e consentimento (regras 1 e 3) |

## 4. O que é medível agora

- Estabilidade do descritor de CM entre sinalizantes: **feito**.
- Coerência do descritor dentro de um grupo de supervisão fraca contra
  grupos diferentes, com a mesma métrica de recuperação.
- Derivação da trajetória e classificação de M: **feito**, com as ressalvas
  da seção 6.2.
- Ruído do descritor por quadro (tremor do MediaPipe em mão parada).

**Não é medível agora:** acurácia de detecção de CL, acurácia de `param`,
separação lexical/CL, atraso real de `param.movimento`, qualquer coisa com
nome de CM, desempenho em sinalização contínua ou com pessoas surdas.

## 5. O que pede decisão

### Linguística (regra 3: equipe linguística e pessoas surdas)

1. **Inventário de configurações de mão** do projeto e como cada grupo sem
   nome do descritor se liga a ele (ou é rejeitado).
2. **Inventário e nomes dos CL** (`CL-*`). O `CL-VEICULO` do exemplo é
   ilustrativo.
3. **Chaves e valores de `param`** por parâmetro: categorias de locação,
   orientação e movimento; se os tipos de movimento de bases públicas servem
   de ponto de partida ou não.
4. Critério de fronteira entre sinal lexical e uso classificador.
5. Construções com as duas mãos (figura e fundo) e como anotá-las.
6. Guia de anotação de trechos de CL.

### Schema (regra 2: versão nova + `spec/CHANGELOG.md`)

1. Fechar as chaves de `param` para `kind: classifier` (v0.1 é livre). Tipos
   de valor: categoria fechada, número contínuo ou ambos.
2. Confiança por parâmetro (`conf` hoje é só do item).
3. Mão do item (dominante/não dominante) e itens manuais sobrepostos no
   tempo, para CL simultâneos.
4. Se `param.movimento` pode chegar depois do item (preenchimento tardio
   em resultado parcial).

Enquanto nada disso estiver decidido, **o núcleo não emite
`kind: classifier`**. Os grupos ficam em artefato de pesquisa.

## 6. Protótipos medidos

Protótipos só em numpy, sem treino de rede. O código e os dados ficam fora
do repositório, no registro do experimento 003. Os números servem para
decidir o próximo passo, não para afirmar desempenho (regra 4).

### 6.1 Descritor de CM

**Dados:** base pública de sinais isolados, com 20 sinais e 12
sinalizantes, gravada em estúdio. Foram usados 1076 clipes e 21.074 quadros
de segurar (mediana de 16 por clipe); 82 clipes não tinham quadro de
segurar. A mão ativa é a que mais se desloca no clipe. **Segurar** quer
dizer que a mão se deslocou menos de 0,05 palma desde o quadro anterior
(causal), com o punho acima do limiar de repouso.

**Métrica:** para cada consulta, os k vizinhos mais próximos entre os itens
de **outros** sinalizantes. P@k é a fração deles com o mesmo número de
sinal. A base de cada consulta é a fração do mesmo sinal na galeria dela,
ou seja, a precisão de escolher ao acaso.

| Variante | Quadro P@1 | P@5 | P@10 | Base | Clipe P@1 | P@5 | Base |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Invariante 3D (proposta) | 0,816 | 0,806 | 0,797 | 0,060 | 0,788 | 0,748 | 0,050 |
| Invariante sem z | 0,814 | 0,805 | 0,796 | 0,060 | 0,782 | 0,744 | 0,050 |
| Pontos crus (punho, palma; sem invariância de rotação e lado) | 0,874 | 0,859 | 0,849 | 0,060 | 0,849 | 0,806 | 0,050 |

A P@1 por quadro, por sinalizante consultado, vai de 0,754 a 0,857 no
invariante e de 0,665 a 0,935 nos pontos crus.

**Achado de método:** na primeira execução, sem o filtro de repouso, dois
sinalizantes tinham 2 a 3 vezes mais quadros de "segurar". Eram mãos paradas
em repouso no início e no fim do clipe, com o punho de 1 a 2 larguras de
ombro abaixo dos ombros. Com elas, a P@1 por quadro caía para 0,48. Mão
parada não é sinal: a detecção de segurar precisa excluir o repouso, e isso
vale também para o branco do CTC.

**O que indica:** quadros parados de um sinal, só pela forma interna da mão,
sem orientação, locação nem lado, acham o mesmo sinal feito por outra pessoa
cerca de 13 vezes mais que o acaso. O descritor carrega informação estável
entre pessoas, que é o pré-requisito para agrupar CM sem nome.

**O que não indica:**
- Que agrupa por **configuração de mão**: o rótulo é o sinal. Sinais
  diferentes com a mesma CM contam como erro. Os pares de sinais
  diferentes mais vizinhos (07-18, 06-09, 11-12, 12-17, 12-13, na numeração
  da base) são candidatos a conferir com a equipe linguística, não
  afirmação.
- Que a invariância é completa na prática: o MediaPipe estima a forma
  pior em certas vistas, então a orientação pode vazar no descritor.
- Que os pontos crus são melhores para CM: eles também carregam orientação
  e lado, que distinguem sinais. A diferença é o que a invariância custa
  **para reconhecer sinal**: 0,07 de P@1 na média, enquanto o pior
  sinalizante sobe 0,05.
- Amostra efetiva: os quadros de um clipe são correlacionados; o tamanho
  real é da ordem de 1076 clipes, 12 pessoas, 20 sinais, estúdio.
- Limiares (0,05 palma, repouso a 1 largura de ombro) vieram de olhar
  poucos clipes, sem calibração.

### 6.2 Movimento

**Dados:** base pública de trajetórias, com 360 curvas 2D de 45 pontos e 15
tipos de movimento, sem identificação de pessoa.

**Derivação:** dá para obter a mesma representação dos nossos landmarks.
Centro dos 21 pontos da mão ativa, do primeiro ao último quadro de segurar,
em larguras de ombro, reamostrado para 45 pontos. Diferenças que ficam: a
base de trajetórias usou o centroide dos pixels da mão segmentada num
"espaço unitário" cujo enquadramento não é documentado (não é min-max por
curva). A amplitude foi casada pela mediana das duas bases (atalho). O
sentido dos eixos e o espelhamento não são documentados, e eles importam
para arcos horário e anti-horário.

**Base de trajetórias, 1-NN euclidiano, acaso 0,067:**

| Validação | Cru | Centrado |
| --- | --- | --- |
| 10 dobras estratificadas ao acaso | 0,872 | 0,875 |
| 4 blocos contíguos de 6 por classe | 0,575 | 0,672 |

A base não identifica a pessoa de cada curva. Se a ordem dentro da classe for
por pessoa, os blocos aproximam teste com pessoa nova; não dá para
verificar. A queda de 0,87 para 0,58–0,67 sugere que a validação ao acaso é
otimista.

**Base de sinais isolados (978 clipes com curva, 12 sinalizantes):** a
classe prevista mais frequente é a 9 (288 clipes), seguida da 5 (167);
classes 11 e 13 quase não aparecem. A classe prevista para um clipe
coincide com a mais comum do mesmo sinal nos **outros** sinalizantes em
0,430 dos clipes, contra 0,274 com os sinais embaralhados.

**O que indica:** a cadeia landmarks → trajetória → tipo de movimento
funciona, e o tipo previsto tem alguma consistência por sinal entre pessoas.

**O que não indica:** que a classe está certa (a base de sinais isolados
não tem rótulo de movimento), nem que os 15 tipos da base de trajetórias
são categorias de movimento de Libras para o projeto. Também há uma
inconsistência na própria base de trajetórias: a classe 9 (zigue-zague
horizontal, pelo nome) tem a menor amplitude média de todas, perto de 0,06
nos dois eixos. A classe 15 (tremor) tem amplitude grande. Nas classes 10 a
12, o eixo de maior amplitude é o contrário do que o nome sugere numa
primeira leitura (pode ser só convenção de nome). Isso deve ir para uma
issue antes de qualquer uso, e não ser corrigido por conta própria. Os
trechos dos sinais isolados são curtos (entre dois segurar), não os cerca
de 7 s de movimento isolado da base de trajetórias. A predominância da
classe de menor amplitude provavelmente vem disso.

## 7. Próximos passos (em ordem)

1. Abrir issue linguística com: inventário de CM, inventário de CL, chaves
   de `param`, uso ou não dos tipos de movimento de bases públicas, e a
   inconsistência das classes 9 e 15 (seção 6.2).
2. Medir o descritor na fonte de supervisão fraca (mesmo grupo contra
   grupos diferentes), com triagem humana dos grupos.
3. Portar o descritor e o detector de segurar para `maosabertas-core`, com
   testes de invariância, sem mexer nas 203 features.
4. Propor a v0.2 do schema só depois das decisões da seção 5.

## Alternativas consideradas

| Alternativa | Por que não (por ora) |
| --- | --- |
| Uma glosa por combinação CL + movimento no vocabulário CTC | Vocabulário explode e não cobre o uso produtivo |
| Modelo único com várias cabeças num ONNX | Quebra o contrato `[T, C]` do ADR-005 |
| Descritor dentro das 203 features já | Muda `FEATURE_DIM` e invalida modelos antes de haver medição do ganho |
| Aprender CM só com rede supervisionada | Não há rótulo de CM; o descritor geométrico mais a supervisão fraca vêm antes |
| Nomear os grupos do descritor | Viola a regra 3 |
