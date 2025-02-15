"""Test the tokenizer module."""
import pytest
import tiktoken
from tokenizers import Tokenizer as HFTokenizer
from transformers import AutoTokenizer

from chonkie.tokenizer import CharacterTokenizer, Tokenizer, WordTokenizer


# Rich test examples
@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """Natural language processing (NLP) is a field of artificial intelligence 
    that focuses on the interaction between computers and human language. It involves 
    tasks like machine translation, text summarization, and sentiment analysis."""

@pytest.fixture
def sample_texts():
    """Sample texts for testing."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Python is a high-level programming language!",
        "Machine learning models require careful evaluation.",
        "Data scientists analyze complex patterns in data.",
        """Neural networks are computational systems inspired by 
        the biological neural networks that constitute animal brains."""
    ]

# Tokenizer fixtures for different backends

# Initialize the WordTokenizer
@pytest.fixture
def word_tokenizer():
    """Word tokenizer fixture."""
    return WordTokenizer()

# Initialize the CharacterTokenizer
@pytest.fixture
def char_tokenizer():
    """Character tokenizer fixture."""
    return CharacterTokenizer()

# Initialize the HuggingFace tokenizer
@pytest.fixture
def hf_tokenizer():
    """Create a HuggingFace tokenizer fixture."""
    return HFTokenizer.from_pretrained("gpt2")

# Initialize the Tiktoken tokenizer
@pytest.fixture
def tiktoken_tokenizer():
    """Create a Tiktoken tokenizer fixture."""
    return tiktoken.get_encoding("gpt2")

# Initialize the Transformer tokenizer
@pytest.fixture
def transformers_tokenizer():
    """Create a Transformer tokenizer fixture."""
    return AutoTokenizer.from_pretrained("gpt2")

# Initialize the Callable tokenizer
@pytest.fixture
def callable_tokenizer():
    """Create a callable tokenizer fixture."""
    return lambda text: len(text.split())

# Test if the Tokenizer class can wrap around the different tokenizer backends
@pytest.mark.parametrize("tokenizer_fixture", [
    "hf_tokenizer",
    "tiktoken_tokenizer",
    "transformers_tokenizer"
])
def test_tokenizer_initialization(request, tokenizer_fixture):
    """Test tokenizer initialization."""
    tokenizer = request.getfixturevalue(tokenizer_fixture)
    tokenizer = Tokenizer(tokenizer)
    assert tokenizer is not None
    assert tokenizer._tokenizer_backend in ["transformers", "tokenizers", "tiktoken"]


# Tests for different tokenizer backends
@pytest.mark.parametrize("tokenizer_fixture", [
    "hf_tokenizer",
    "tiktoken_tokenizer",
    "transformers_tokenizer"
])
def test_tokenizer_encode_decode(request, tokenizer_fixture, sample_text):
    """Test encoding and decoding across different backends."""
    try:
        tokenizer = request.getfixturevalue(tokenizer_fixture)
        tokenizer = Tokenizer(tokenizer)
    except Exception as e:
        pytest.skip(f"Skipping due to missing backend: {str(e)}")

    # Test encode
    tokens = tokenizer.encode(sample_text)
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    assert all(isinstance(token, int) for token in tokens)
    
    # Test decode
    if tokenizer._tokenizer_backend != "callable":
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)
        assert decoded == sample_text
        
@pytest.mark.parametrize("tokenizer_fixture", [
    "hf_tokenizer",
    "tiktoken_tokenizer",
    "transformers_tokenizer"
])
def test_tokenizer_batch_operations(request, tokenizer_fixture, sample_texts):
    """Test batch encoding and decoding across different backends."""
    try:
        tokenizer = request.getfixturevalue(tokenizer_fixture)
        tokenizer = Tokenizer(tokenizer)
    except Exception as e:
        pytest.skip(f"Skipping due to missing backend: {str(e)}")

    # Test batch encode
    token_lists = tokenizer.encode_batch(sample_texts)
    assert isinstance(token_lists, list)
    assert len(token_lists) == len(sample_texts)
    assert all(isinstance(tokens, list) for tokens in token_lists)
    assert all(len(tokens) > 0 for tokens in token_lists)
    assert all(all(isinstance(token, int) for token in tokens) for tokens in token_lists)
    
    # Test batch decode
    if tokenizer._tokenizer_backend != "callable":
        decoded_texts = tokenizer.decode_batch(token_lists)
        assert isinstance(decoded_texts, list)
        assert len(decoded_texts) == len(sample_texts)
        assert all(isinstance(text, str) for text in decoded_texts)
        assert decoded_texts == sample_texts

@pytest.mark.parametrize("tokenizer_fixture", [
    "hf_tokenizer",
    "tiktoken_tokenizer",
    "transformers_tokenizer",
    "callable_tokenizer"
])
def test_tokenizer_token_counts(request, tokenizer_fixture, sample_text, sample_texts):
    """Test token counting across different backends."""
    try:
        tokenizer = request.getfixturevalue(tokenizer_fixture)
        tokenizer = Tokenizer(tokenizer)
    except Exception as e:
        pytest.skip(f"Skipping due to missing backend: {str(e)}")

    # Test single text token count
    count = tokenizer.count_tokens(sample_text)
    assert isinstance(count, int)
    assert count > 0
    
    # Verify count matches encoded length
    if tokenizer._tokenizer_backend != "callable":
        assert count == len(tokenizer.encode(sample_text))
    
    # Test batch token count
    counts = tokenizer.count_tokens_batch(sample_texts)
    assert isinstance(counts, list)
    assert len(counts) == len(sample_texts)
    assert all(isinstance(c, int) for c in counts)
    assert all(c > 0 for c in counts)
    
    # Verify counts match encoded lengths
    if tokenizer._tokenizer_backend != "callable":
        encoded_lengths = [len(tokens) for tokens in tokenizer.encode_batch(sample_texts)]
        assert counts == encoded_lengths

@pytest.mark.parametrize("tokenizer_fixture", [
    "hf_tokenizer",
    "tiktoken_tokenizer",
    "transformers_tokenizer",
    "callable_tokenizer"
])
def test_tokenizer_backend_detection(request, tokenizer_fixture):
    """Test that the tokenizer correctly identifies its backend."""
    try:
        tokenizer = request.getfixturevalue(tokenizer_fixture)
        tokenizer = Tokenizer(tokenizer)
    except Exception as e:
        pytest.skip(f"Skipping due to missing backend: {str(e)}")

    assert tokenizer._tokenizer_backend in ["transformers", "tokenizers", "tiktoken", "callable"]
    
def test_tokenizer_error_handling():
    """Test error handling for unsupported operations."""
    # Test with callable tokenizer
    def dummy_counter(text): return len(text.split())
    tokenizer = Tokenizer(dummy_counter)
    
    # Should work
    assert tokenizer.count_tokens("This is a test") == 4
    
    # Should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        tokenizer.encode("This should fail")
    
    with pytest.raises(NotImplementedError):
        tokenizer.decode([1, 2, 3])
    
    with pytest.raises(NotImplementedError):
        tokenizer.encode_batch(["This", "should", "fail"])
    
    # Test with invalid backend
    with pytest.raises(ValueError):
        Tokenizer(object())  # Should raise error for unsupported tokenizer type

# Tests for WordTokenizer
def test_word_tokenizer_initialization(word_tokenizer):
    """Test WordTokenizer initialization."""
    assert word_tokenizer.vocab == [' ']
    assert len(word_tokenizer.token2id) == 1
    assert word_tokenizer.token2id[' '] == 0

def test_word_tokenizer_encode_decode(word_tokenizer, sample_text):
    """Test encoding and decoding with WordTokenizer."""
    tokens = word_tokenizer.encode(sample_text)
    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)
    
    decoded = word_tokenizer.decode(tokens)
    assert isinstance(decoded, str)
    assert decoded.strip() == sample_text.strip()

def test_word_tokenizer_batch_operations(word_tokenizer, sample_texts):
    """Test batch operations with WordTokenizer."""
    token_lists = word_tokenizer.encode_batch(sample_texts)
    assert isinstance(token_lists, list)
    assert all(isinstance(tokens, list) for tokens in token_lists)
    
    decoded_texts = word_tokenizer.decode_batch(token_lists)
    assert isinstance(decoded_texts, list)
    assert all(isinstance(text, str) for text in decoded_texts)
    assert [text.strip() for text in decoded_texts] == [text.strip() for text in sample_texts]

def test_word_tokenizer_vocabulary_growth(word_tokenizer):
    """Test that vocabulary grows correctly with new words."""
    initial_vocab_size = len(word_tokenizer.vocab)
    word_tokenizer.encode("hello world")
    assert len(word_tokenizer.vocab) > initial_vocab_size
    assert "hello" in word_tokenizer.vocab
    assert "world" in word_tokenizer.vocab

# Tests for CharacterTokenizer
def test_char_tokenizer_initialization(char_tokenizer):
    """Test CharacterTokenizer initialization."""
    assert char_tokenizer.vocab == [' ']
    assert len(char_tokenizer.token2id) == 1
    assert char_tokenizer.token2id[' '] == 0

def test_char_tokenizer_encode_decode(char_tokenizer, sample_text):
    """Test encoding and decoding with CharacterTokenizer."""
    tokens = char_tokenizer.encode(sample_text)
    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)
    assert len(tokens) == len(sample_text)
    
    decoded = char_tokenizer.decode(tokens)
    assert isinstance(decoded, str)
    assert decoded == sample_text

def test_char_tokenizer_batch_operations(char_tokenizer, sample_texts):
    """Test batch operations with CharacterTokenizer."""
    token_lists = char_tokenizer.encode_batch(sample_texts)
    assert isinstance(token_lists, list)
    assert all(isinstance(tokens, list) for tokens in token_lists)
    assert all(len(tokens) == len(text) for tokens, text in zip(token_lists, sample_texts))
    
    decoded_texts = char_tokenizer.decode_batch(token_lists)
    assert isinstance(decoded_texts, list)
    assert all(isinstance(text, str) for text in decoded_texts)
    assert decoded_texts == sample_texts

def test_char_tokenizer_count_tokens(char_tokenizer, sample_text, sample_texts):
    """Test token counting with CharacterTokenizer."""
    count = char_tokenizer.count_tokens(sample_text)
    assert count == len(sample_text)
    
    counts = char_tokenizer.count_tokens_batch(sample_texts)
    assert counts == [len(text) for text in sample_texts]

def test_tokenizer_string_representation():
    """Test string representation of tokenizers."""
    char_tokenizer = CharacterTokenizer()
    word_tokenizer = WordTokenizer()
    
    assert str(char_tokenizer) == "CharacterTokenizer(vocab_size=1)"
    assert str(word_tokenizer) == "WordTokenizer(vocab_size=1)"