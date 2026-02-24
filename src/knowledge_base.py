import re
from typing import List, Dict

class ClinicalKnowledgeBase:
    def __init__(self, knowledge_book_path="clinical_knowledge_book.md"):
        self.kb_path = knowledge_book_path
        self.chunks: List[Dict[str, str]] = []
        self.index: Dict[str, List[int]] = {} # keyword -> list of chunk indices
        
    def load(self):
        """
        Parses the markdown file into semantic chunks based on headers.
        """
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Split by Level 2 Headers (##)
            # This logic assumes the structure used in clinical_knowledge_book.md
            # We want to capture the header and the content following it
            sections = re.split(r'(^##\s+.*$)', content, flags=re.MULTILINE)
            
            current_chunk = {"header": "Introduction", "content": ""}
            
            # sections[0] is intro text before first header
            if sections[0].strip():
                current_chunk["content"] = sections[0].strip()
                self.chunks.append(current_chunk)
                
            # Iterate over header-content pairs
            for i in range(1, len(sections), 2):
                header = sections[i].strip().replace("##", "").strip()
                body = sections[i+1].strip() if i+1 < len(sections) else ""
                
                # Further split by Level 3 Headers (###) if present
                subsections = re.split(r'(^###\s+.*$)', body, flags=re.MULTILINE)
                
                # Main body of level 2 (before first h3)
                if subsections[0].strip():
                    self.chunks.append({
                        "header": header,
                        "content": subsections[0].strip()
                    })
                    
                for j in range(1, len(subsections), 2):
                    sub_header = subsections[j].strip().replace("###", "").strip()
                    sub_body = subsections[j+1].strip()
                    
                    self.chunks.append({
                        "header": f"{header} > {sub_header}",
                        "content": sub_body
                    })
                    
            print(f"Knowledge Base Loaded. Indexed {len(self.chunks)} clinical protocols.")
            self._build_keyword_index()
            
        except FileNotFoundError:
            print(f"Warning: Knowledge Book not found at {self.kb_path}")
            
    def _build_keyword_index(self):
        """
        Simple inverted index for keyword search.
        """
        stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with"}
        
        for idx, chunk in enumerate(self.chunks):
            text = (chunk["header"] + " " + chunk["content"]).lower()
            # Tokenize by non-alphanumeric
            tokens = re.split(r'[^a-z0-9]+', text)
            
            for token in tokens:
                if len(token) > 2 and token not in stop_words:
                    if token not in self.index:
                        self.index[token] = []
                    if idx not in self.index[token]:
                        self.index[token].append(idx)

    def retrieve(self, query: str, top_k=3) -> str:
        """
        Returns relevant chunks as a single context string.
        """
        query_tokens = [t.lower() for t in re.split(r'[^a-zA-Z0-9]+', query) if len(t) > 2]
        
        # simple scoring: term frequency in query * presence in document
        # (Since query is short, unweighted boolean match count is effective enough)
        
        scores: Dict[int, int] = {}
        
        for token in query_tokens:
            if token in self.index:
                for chunk_idx in self.index[token]:
                    scores[chunk_idx] = scores.get(chunk_idx, 0) + 1
                    
        # Sort by score desc
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take top K
        results = []
        for idx, score in sorted_chunks[:top_k]:
            chunk = self.chunks[idx]
            results.append(f"--- PROTOCOL: {chunk['header']} ---\n{chunk['content']}")
            
        if not results:
            return "No specific clinical protocols found."
            
        return "\n\n".join(results)

if __name__ == "__main__":
    # Test
    kb = ClinicalKnowledgeBase("c:/Users/hp/Downloads/MedGamma/clinical_knowledge_book.md")
    kb.load()
    print("\n--- Search: 'nodule' ---")
    print(kb.retrieve("lung nodule ground glass"))
