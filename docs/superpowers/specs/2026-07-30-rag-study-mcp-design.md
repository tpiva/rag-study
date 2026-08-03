# RAG Study — servidor MCP de busca sobre documentos pessoais

## Objetivo

Projeto de estudo para atualizar conhecimentos de Python e aprender RAG (Retrieval-Augmented Generation) na prática, construindo uma ferramenta pessoal útil: um servidor MCP que permite consultar, via Claude (Desktop e Claude Code), uma coleção de documentos pessoais — majoritariamente PDFs — guardados numa pasta local.

O Claude que o usuário já paga por assinatura (Desktop / Claude Code) é o consumidor da ferramenta. Nenhuma chamada à API paga da Anthropic acontece dentro do código do projeto — a consulta principal (perguntar sobre os documentos) passa inteiramente pelo protocolo MCP, sem custo de API.

## Arquitetura

```
<PASTA EXTERNA CONFIGURÁVEL>/   # PDFs e outros documentos do usuário (fora do projeto)
        │
        ▼
rag-study/
├── docs/
│   └── notes/               # notas de estudo por etapa, escritas manualmente durante o desenvolvimento
├── src/
│   ├── ingest.py             # CLI: (re)indexa os documentos da pasta externa
│   ├── mcp_server.py         # servidor MCP: expõe a tool de busca semântica
│   └── config.py             # caminhos, modelo de embedding, chunk size etc.
├── storage/                  # índice ChromaDB persistido (gitignored)
├── pyproject.toml            # gerenciado com uv
└── README.md
```

Fluxo:

1. O usuário roda `uv run src/ingest.py` sempre que adicionar/alterar documentos na pasta externa.
2. O script extrai texto, quebra em chunks, gera embeddings localmente (HuggingFace) e grava/atualiza o índice ChromaDB em `storage/`.
3. No Claude Desktop ou Claude Code, o usuário pergunta algo sobre seus documentos.
4. O Claude decide chamar a tool `buscar_documentos` do servidor MCP.
5. O servidor MCP consulta o ChromaDB via LlamaIndex e retorna os trechos mais relevantes, com a fonte (nome do arquivo + página).
6. O Claude usa esses trechos para responder, citando de qual documento veio a informação.

## Decisões técnicas

| Decisão | Escolha | Motivo |
|---|---|---|
| Framework de RAG | LlamaIndex | Criado especificamente para RAG, API mais direta para o caso de uso (indexar + consultar documentos) |
| Embeddings | Modelo local via HuggingFace (`HuggingFaceEmbedding`) | 100% gratuito, offline, sem chave de API, documentos não saem da máquina |
| Vector store | ChromaDB (persistido em disco local) | Gratuito, sem servidor externo, integração direta com LlamaIndex, amplamente usado no mercado |
| Gerenciador de pacotes/ambiente | `uv` | Ferramenta moderna e rápida, boa oportunidade de atualizar o conhecimento de tooling Python |
| Localização dos documentos | Pasta externa configurável (não dentro do projeto) | Os documentos já existem em outro lugar do disco do usuário |
| Interface MCP | SDK oficial `mcp` (estilo FastMCP), configurado tanto no Claude Desktop quanto no Claude Code | Único ponto de consumo do RAG, sem custo de API |
| Chamadas à API paga da Anthropic | Nenhuma no código do projeto | Requisito explícito do usuário — o Claude já pago via assinatura é quem consome a tool MCP |

## Componentes

- **`config.py`** — centraliza caminhos (pasta externa de documentos, diretório de storage do ChromaDB), parâmetros de chunking e nome do modelo de embedding. Evita "magic values" espalhados pelo código.
- **`ingest.py`** — usa `SimpleDirectoryReader` do LlamaIndex para ler os documentos da pasta configurada, quebra em chunks com um `NodeParser`, gera embeddings via `HuggingFaceEmbedding` e persiste no ChromaDB. Rodado manualmente pelo usuário.
- **`mcp_server.py`** — usa o SDK oficial `mcp` para expor a tool `buscar_documentos(pergunta, top_k=5)`, que carrega o índice existente no ChromaDB, faz a busca semântica via LlamaIndex e retorna os trechos relevantes com a fonte. Processo de longa duração, configurado nos dois clientes Claude.

## Tratamento de erros

- Documentos corrompidos ou sem texto extraível (ex: PDF escaneado sem OCR) são pulados durante a ingestão com um aviso no console — não interrompem o processo inteiro.
- Se o índice em `storage/` ainda não existir quando o servidor MCP subir, a tool `buscar_documentos` retorna uma mensagem clara pedindo para rodar a ingestão primeiro, em vez de quebrar.

## Notas de estudo

Cada fase de desenvolvimento (extração de documentos, chunking, embeddings, vector store, servidor MCP, integração com os clientes) ganha uma nota curta em `docs/notes/`, escrita manualmente (pelo usuário ou pelo Claude Code durante a implementação) explicando o conceito por trás e as decisões tomadas. Serve tanto de registro do aprendizado quanto de referência futura. Não há script automatizado de geração de notas — isso foi avaliado e descartado para não introduzir uma chamada de API paga desnecessária no projeto.

## Fases de desenvolvimento

| Fase | Entrega | Aprendizado-chave |
|---|---|---|
| 0 | Setup do projeto com `uv`, `pyproject.toml`, estrutura de pastas | Ferramental Python moderno |
| 1 | Leitura da pasta externa configurável + extração de texto dos documentos | Manipulação de arquivos, extração de PDFs em Python |
| 2 | Chunking dos textos extraídos | Estratégias de divisão de texto para RAG |
| 3 | Geração de embeddings locais (HuggingFace) | Embeddings, modelos locais |
| 4 | Indexação no ChromaDB + script de ingestão completo (CLI) | Vector stores, scripts CLI |
| 5 | Servidor MCP com a tool de busca semântica | Protocolo MCP, servidores em Python |
| 6 | Configuração no Claude Desktop e no Claude Code | Integração de cliente MCP |
| 7 (futuro) | Melhorias: OCR para PDFs escaneados, reindexação incremental, watch de pasta | Extensões pós-MVP, fora do escopo inicial |

## Fora de escopo (v1)

- OCR de documentos escaneados sem camada de texto.
- Reindexação automática/incremental (watch da pasta) — a ingestão é manual via CLI.
- Qualquer chamada à API paga da Anthropic dentro do código do projeto.
- Suporte a múltiplos usuários/multi-tenancy — projeto de uso pessoal, single-user.
