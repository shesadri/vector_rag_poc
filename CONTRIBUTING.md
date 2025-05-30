# Contributing to Vector RAG POC

Thank you for your interest in contributing to the Vector RAG POC project! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose
- Git

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/vector_rag_poc.git
   cd vector_rag_poc
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If you create this for dev dependencies
   ```

4. **Setup Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Start Elasticsearch**
   ```bash
   docker-compose up -d elasticsearch
   ```

6. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker
- Include clear description of the problem
- Provide steps to reproduce
- Include system information (OS, Python version, etc.)
- Add relevant logs or error messages

### Suggesting Features

- Check existing issues first
- Provide clear use case and benefits
- Include implementation suggestions if possible
- Consider backwards compatibility

### Code Contributions

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   pytest tests/ -v
   python -m pytest tests/test_search.py::TestVectorRAG::test_health_endpoint
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new embedding model support"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use meaningful variable and function names

### Code Structure

```python
# Good example
from typing import List, Dict, Any, Optional
from loguru import logger

def generate_embeddings(texts: List[str], 
                       model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        model_name: Name of the embedding model to use
    
    Returns:
        List of embedding vectors
    
    Raises:
        ValueError: If texts list is empty
    """
    if not texts:
        raise ValueError("texts list cannot be empty")
    
    logger.info(f"Generating embeddings for {len(texts)} texts")
    # Implementation here
    return embeddings
```

### Documentation

- Use clear docstrings for all functions and classes
- Include type information in docstrings
- Update README.md for significant changes
- Add examples for new features

### Testing

- Write tests for all new functionality
- Aim for good test coverage
- Use descriptive test names
- Include both positive and negative test cases

```python
def test_embedding_generation_with_valid_input():
    """Test that embeddings are generated correctly for valid input"""
    texts = ["test text 1", "test text 2"]
    embeddings = generate_embeddings(texts)
    
    assert len(embeddings) == len(texts)
    assert all(isinstance(emb, list) for emb in embeddings)
    assert all(len(emb) == EXPECTED_DIMENSION for emb in embeddings)

def test_embedding_generation_with_empty_input():
    """Test that appropriate error is raised for empty input"""
    with pytest.raises(ValueError, match="texts list cannot be empty"):
        generate_embeddings([])
```

## Areas for Contribution

### High Priority

1. **Additional Embedding Models**
   - Support for OpenAI embeddings
   - Hugging Face model integration
   - Multi-language models

2. **Search Improvements**
   - Better hybrid search algorithms
   - Query expansion techniques
   - Re-ranking strategies

3. **Performance Optimization**
   - Batch processing improvements
   - Caching strategies
   - Memory optimization

### Medium Priority

1. **Documentation Enhancements**
   - More usage examples
   - Performance tuning guide
   - Deployment instructions

2. **Testing and Quality**
   - Integration tests
   - Performance benchmarks
   - Load testing

3. **Monitoring and Observability**
   - Metrics collection
   - Health check improvements
   - Logging enhancements

### Future Enhancements

1. **Advanced Features**
   - Document chunking strategies
   - Multi-modal embeddings (text + images)
   - Federated search across multiple indices

2. **UI Components**
   - Web interface for document management
   - Search result visualization
   - Analytics dashboard

3. **Integrations**
   - LLM provider integrations
   - Document processing pipelines
   - External data source connectors

## Pull Request Guidelines

### PR Description Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. All discussions resolved
4. Up-to-date with main branch

## Development Guidelines

### Adding New Embedding Models

```python
# Example: Adding OpenAI embeddings support
class OpenAIEmbeddings(BaseEmbeddings):
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def generate_embedding(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
```

### Adding New Search Features

```python
# Example: Adding semantic filtering
def semantic_filter_search(self, query_embedding: List[float],
                          semantic_filters: Dict[str, str]) -> List[Dict]:
    """Search with semantic filtering on metadata fields"""
    # Implementation here
    pass
```

## Release Process

1. Version follows semantic versioning (MAJOR.MINOR.PATCH)
2. Update CHANGELOG.md
3. Tag release in Git
4. Update documentation

## Getting Help

- Check existing issues and documentation
- Join discussions in GitHub Issues
- Ask questions in Pull Request comments
- Contact maintainers for complex questions

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes for significant contributions
- Invited to become maintainers based on consistent contributions

Thank you for contributing to Vector RAG POC!