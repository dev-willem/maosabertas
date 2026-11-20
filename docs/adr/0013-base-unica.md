# ADR-013: Base única de dados em SQLite

- **Status:** aceita
- **Data:** 2026-11-20

## Contexto

Cada fonte de dados ficava numa pasta com formato próprio: manifestos CSV e `.npz`
por clipe, legendas em JSON, zips de bases públicas com sensores diferentes. Juntar
fontes exigia `corpus.merge_manifests` e caminhos absolutos; licença e origem do
rótulo ficavam em arquivos de texto soltos. O responsável pelo projeto pediu uma base
só, "um SQLite ou algo mais simples por enquanto".

## Decisão

1. **Um arquivo SQLite** (`sqlite3` da biblioteca padrão do Python, sem dependência
   nova) com quatro tabelas:

   | Tabela | O que guarda |
   | --- | --- |
   | `fonte` | nome, licença, se permite uso comercial, crédito |
   | `sinalizante` | pseudônimo e a fonte a que pertence (nunca o nome) |
   | `amostra` | os bytes do `.npz` extraído, como vieram (comprimidos), com sha256, formato do sensor e número de quadros |
   | `rotulo` | trilhas como no ELAN: `glosa`, `traducao`, `...-fonte`, com tempo opcional e a **origem** do rótulo |

2. **Rótulo fraco não se mistura:** `origem` diz se o rótulo veio de anotação, da
   fala do vídeo ou de texto na imagem. Rótulo no vocabulário da própria fonte fica
   em trilha `...-fonte` e não vira categoria do projeto (regra 3).
3. **Licença viaja com o dado:** o treino com `--base` grava no experimento as
   fontes usadas, as licenças e `comercial: false` se qualquer fonte proibir uso
   comercial.
4. **Integridade:** reimportar é idempotente (mesma chave e mesmo sha256); conteúdo
   diferente na mesma chave, licença divergente ou o mesmo pseudônimo em duas fontes
   são erro. Esse último evitaria que duas pessoas virassem uma na divisão por
   sinalizante.
5. **Privacidade:** a base tem landmarks de pessoas reais e fica fora do repositório
   (o módulo recusa caminho dentro dele). `.sqlite` e `.db` entram no `.gitignore` e
   no job "Nenhum dado pessoal no commit" do CI. Vídeo nunca entra.
6. **O código no git é genérico** (`maosabertas_training.base`): importa manifestos
   e oferece `inserir` para qualquer formato. Os conversores de cada fonte nomeiam a
   fonte e ficam locais, fora do git.
7. O treino lê da base (`train --base BASE --fontes ...`) pelo mesmo caminho de
   features do tempo real (ADR-004). O manifesto continua aceito.

## Alternativas consideradas

| Alternativa | Por que não (por ora) |
| --- | --- |
| Continuar com pastas e manifestos | É o problema que motivou a decisão |
| Parquet ou HDF5 | Dependência nova e sem transações nem restrições de integridade |
| PostgreSQL (o banco previsto no stack) | Servidor para rodar e manter, num treino offline e de uma pessoa; o esquema migra quando houver servidor |
| Landmarks em colunas (um valor por linha) | Arquivo dezenas de vezes maior e lento; o treino lê a amostra inteira de uma vez |
| Guardar só o caminho do `.npz` | Não seria uma base só: continuaria dependendo das pastas |

## Consequências

- Um arquivo para fazer cópia de segurança, consultar por fonte, glosa ou
  sinalizante, e treinar.
- O SQLite não é para escrita concorrente: um importador por vez.
- As pastas originais não são apagadas pela migração; a base pode ser conferida
  contra elas (sha256) antes de qualquer limpeza, que é decisão do responsável.
- Mudança de esquema sobe `PRAGMA user_version`; o código recusa versão que não
  conhece.
