from typing import Dict, Any, List


class ContextBuilder:
    """Builds context text from ChromaDB query results."""

    @staticmethod
    def prepare_context(query_results: Dict[str, Any]) -> str:
        """
        Prepare context text from ChromaDB query results.

        Args:
            query_results: Dictionary from ChromaDB query

        Returns:
            Formatted context text
        """
        documents = query_results['documents'][0]
        metadatas = query_results['metadatas'][0]

        context_parts = []

        for i, (doc, metadata) in enumerate(zip(documents, metadatas), 1):
            context_parts.append(f"[{i}]")
            context_parts.append(f"Title: {metadata['title']}")
            context_parts.append(f"Type: {metadata['type']}")
            context_parts.append(f"Category: {metadata['category']}")
            context_parts.append(f"Content: {doc}")
            context_parts.append("")  # Empty line between documents

        return "\n".join(context_parts)

    @staticmethod
    def extract_source_documents(query_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract source document metadata from query results.

        Args:
            query_results: Dictionary from ChromaDB query

        Returns:
            List of source document dictionaries
        """
        metadatas = query_results['metadatas'][0]
        distances = query_results['distances'][0]
        documents = query_results['documents'][0]

        source_documents = []
        for metadata, distance, document in zip(metadatas, distances, documents):
            source_documents.append({
                'title': metadata.get('title', 'Untitled policy'),
                'type': metadata.get('type', 'Policy'),
                'category': metadata.get('category', 'University of Florida'),
                'distance': float(distance),
                'excerpt': (document or '').strip()[:600]
            })

        return source_documents
