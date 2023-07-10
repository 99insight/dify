import logging

from flask import current_app
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import BaseTool

from langchain.schema import Document

from core.callback_handler.index_tool_callback_handler import DatasetIndexToolCallbackHandler
from core.embedding.cached_embedding import CacheEmbedding
from core.index.keyword_table_index.keyword_table_index import KeywordTableIndex, KeywordTableConfig
from core.index.vector_index.vector_index import VectorIndex
from core.llm.llm_builder import LLMBuilder
from models.dataset import Dataset


class DatasetTool(BaseTool):
    """Tool for querying a Dataset."""

    dataset: Dataset
    k: int = 2

    def _run(self, tool_input: str) -> str:
        if self.dataset.indexing_technique == "economy":
            # use keyword table query
            kw_table_index = KeywordTableIndex(
                dataset=self.dataset,
                config=KeywordTableConfig(
                    max_keywords_per_chunk=5
                )
            )

            documents = kw_table_index.search(tool_input, search_kwargs={'k': self.k})
        else:
            model_credentials = LLMBuilder.get_model_credentials(
                tenant_id=self.dataset.tenant_id,
                model_provider=LLMBuilder.get_default_provider(self.dataset.tenant_id),
                model_name='text-embedding-ada-002'
            )

            embeddings = CacheEmbedding(OpenAIEmbeddings(
                **model_credentials
            ))

            vector_index = VectorIndex(
                dataset=self.dataset,
                config=current_app.config,
                embeddings=embeddings
            )

            documents_height = vector_index.search(
                tool_input,
                search_type='similarity_score_threshold',
                search_kwargs={
                    'k': self.k
                }
            )

            kw_table_indexs = KeywordTableIndex(
                dataset=self.dataset,
                config=KeywordTableConfig(
                    max_keywords_per_chunk=5
                )
            )

            documents_economy = kw_table_indexs.search(tool_input, search_kwargs={'k': self.k})

            documents_es=[]
            for kw_table in documents_economy:
                documents_es.append(Document(
                    page_content=kw_table.page_content,
                    metadata={
                        "doc_id": kw_table.metadata.get('doc_id'),
                        "score": kw_table.metadata.get('score')
                    }
                ))
            documents_hs=[]
            for vc_table in documents_height:
                documents_hs.append(Document(
                    page_content=vc_table.page_content,
                    metadata={
                        "doc_id": vc_table.metadata.get('doc_id'),
                        "score": vc_table.metadata.get('score')
                    }
                ))

            documents=[]
            for document_h in documents_hs:
                documents.append(Document(
                    page_content=document_h.page_content,
                    metadata={
                        "doc_id": document_h.metadata.get('doc_id'),
                        "score": document_h.metadata.get('score')*0.6+0.4*[document_e.metadata.get('score') if str(document_e.metadata.get('doc_id'))==str(document_h.metadata.get('doc_id')) else 0 for document_e in documents_es][0]
                    }
                ))

            for document_e in documents_es:
                if document_e.metadata.get('doc_id') not in [doc_id.metadata.get('doc_id') for doc_id in documents]:
                    documents.append(Document(
                        page_content=document_e.page_content,
                        metadata={
                            "doc_id": document_e.metadata.get('doc_id'),
                            "score": 0.4 * document_e.metadata.get('score')
                        }
                    ))

            documents = sorted(
                documents,
                key=lambda x: x.metadata.get('score'),
                reverse=True,
            )

            hit_callback = DatasetIndexToolCallbackHandler(self.dataset.id)
            hit_callback.on_tool_end(documents)
        return str("\n".join([document.page_content for document in documents]))

    async def _arun(self, tool_input: str) -> str:
        model_credentials = LLMBuilder.get_model_credentials(
            tenant_id=self.dataset.tenant_id,
            model_provider=LLMBuilder.get_default_provider(self.dataset.tenant_id),
            model_name='text-embedding-ada-002'
        )

        embeddings = CacheEmbedding(OpenAIEmbeddings(
            **model_credentials
        ))

        vector_index = VectorIndex(
            dataset=self.dataset,
            config=current_app.config,
            embeddings=embeddings
        )

        documents = await vector_index.asearch(
            tool_input,
            search_type='similarity',
            search_kwargs={
                'k': 10
            }
        )

        hit_callback = DatasetIndexToolCallbackHandler(self.dataset.id)
        hit_callback.on_tool_end(documents)
        return str("\n".join([document.page_content for document in documents]))
