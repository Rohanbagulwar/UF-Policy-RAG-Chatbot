from typing import Dict, Any, Optional

from .source_catalog import PolicySourceCatalog


class RetrievalService:
    """Main service orchestrating the retrieval pipeline."""

    def __init__(
        self,
        embedding_model,
        chroma_manager,
        openai_manager,
        context_builder,
        prompt_builder,
        source_catalog: Optional[PolicySourceCatalog] = None
    ):
        """
        Initialize retrieval service with all components.

        Args:
            embedding_model: EmbeddingModel instance
            chroma_manager: ChromaDBManager instance
            openai_manager: OpenAIClientManager instance
            context_builder: ContextBuilder instance
            prompt_builder: PromptBuilder instance
        """
        self.embedding_model = embedding_model
        self.chroma_manager = chroma_manager
        self.openai_manager = openai_manager
        self.context_builder = context_builder
        self.prompt_builder = prompt_builder
        self.source_catalog = source_catalog or PolicySourceCatalog()

    def query(
        self,
        question: str,
        n_results: int = 3,
        filter_by_type: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute complete retrieval and generation pipeline.

        Args:
            question: User's question
            n_results: Number of documents to retrieve
            filter_by_type: Optional filter by document type
            model: OpenAI model name (optional)
            temperature: Temperature for generation (optional)
            max_tokens: Max tokens for generation (optional)

        Returns:
            Dictionary with answer and metadata
        """
        # 1. Create query embedding
        query_embedding = self.embedding_model.encode([question])[0].tolist()

        # 2. Query ChromaDB
        where_filter = None
        if filter_by_type:
            where_filter = {"type": {"$eq": filter_by_type}}

        query_results = self.chroma_manager.query(
            query_embedding=query_embedding,
            n_results=n_results,
            where_filter=where_filter
        )
        sources = self.context_builder.extract_source_documents(query_results)
        for source in sources:
            resolved_source = self.source_catalog.get_source(source['title'])
            if resolved_source:
                indexed_title = source['title']
                source['title'], source['link'] = resolved_source
                if source['title'] != indexed_title:
                    source['notice'] = (
                        "This passage uses a retired policy title. The linked page is "
                        "the current UF policy and its wording may have changed."
                    )
            else:
                source['link'] = None

        # 3. Build context
        context = self.context_builder.prepare_context(query_results)

        # 4. Build messages
        messages = self.prompt_builder.build_messages(question, context)

        # 5. Query OpenAI
        response = self.openai_manager.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # 6. Prepare result
        return {
            'question': question,
            'answer': response.choices[0].message.content,
            'sources': sources,
            'model': model or self.openai_manager.config.model,
            'temperature': temperature or self.openai_manager.config.temperature,
            'tokens_used': {
                'prompt': response.usage.prompt_tokens,
                'completion': response.usage.completion_tokens,
                'total': response.usage.total_tokens
            }
        }
