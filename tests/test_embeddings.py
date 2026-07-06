import json
import urllib.error
from unittest.mock import MagicMock, patch
import pytest

from bayes_brain.embeddings import GeminiEmbedder, OpenAIEmbedder


def test_gemini_embedder_missing_key():
    with patch.dict("os.environ", {}, clear=True):
        embedder = GeminiEmbedder(api_key=None)
        with pytest.raises(ValueError, match="Gemini API key is required"):
            embedder.embed_query("test text")


@patch("urllib.request.urlopen")
def test_gemini_embedder_rest_success(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "embedding": {
            "values": [0.1, 0.2, 0.3]
        }
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    embedder = GeminiEmbedder(api_key="fake-gemini-key", model_name="text-embedding-004")
    result = embedder.embed_query("hello")

    assert result == [0.1, 0.2, 0.3]

    # Verify correct url request is constructed
    args, kwargs = mock_urlopen.call_args
    req = args[0]
    assert req.full_url == "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=fake-gemini-key"
    assert req.method == "POST"
    
    # Check body
    body_data = json.loads(req.data.decode("utf-8"))
    assert body_data == {
        "content": {
            "parts": [{"text": "hello"}]
        }
    }


@patch("urllib.request.urlopen")
def test_gemini_embedder_rest_http_error(mock_urlopen):
    # Simulate urllib HTTPError
    mock_fp = MagicMock()
    mock_fp.read.return_value = b"Invalid API key or model"
    error = urllib.error.HTTPError(
        url="http://fake", code=400, msg="Bad Request", hdrs=None, fp=mock_fp
    )
    mock_urlopen.side_effect = error

    embedder = GeminiEmbedder(api_key="fake-key")
    with pytest.raises(RuntimeError, match="Gemini API request failed with status 400: Invalid API key or model"):
        embedder.embed_query("hello")


def test_gemini_embedder_new_sdk_client():
    # Test new google-genai style client
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.embedding.values = [0.9, 0.8, 0.7]
    mock_client.models.embed_content.return_value = mock_resp

    embedder = GeminiEmbedder(client=mock_client, model_name="custom-model")
    result = embedder.embed_query("hello SDK")

    assert result == [0.9, 0.8, 0.7]
    mock_client.models.embed_content.assert_called_once_with(
        model="custom-model", contents="hello SDK"
    )


def test_gemini_embedder_legacy_sdk_client_dict():
    # Test legacy google-generativeai style returning dict
    mock_client = MagicMock()
    # Delete .models attribute so it doesn't try new SDK path
    del mock_client.models
    mock_client.embed_content.return_value = {
        "embedding": [0.4, 0.5, 0.6]
    }

    embedder = GeminiEmbedder(client=mock_client, model_name="custom-model")
    result = embedder.embed_query("hello legacy dict")

    assert result == [0.4, 0.5, 0.6]
    mock_client.embed_content.assert_called_once_with(
        model="custom-model", contents="hello legacy dict"
    )


def test_gemini_embedder_legacy_sdk_client_values_dict():
    # Test legacy google-generativeai style returning dict with values key
    mock_client = MagicMock()
    del mock_client.models
    mock_client.embed_content.return_value = {
        "embedding": {"values": [0.4, 0.5, 0.6]}
    }

    embedder = GeminiEmbedder(client=mock_client, model_name="custom-model")
    result = embedder.embed_query("hello legacy dict values")

    assert result == [0.4, 0.5, 0.6]


def test_gemini_embedder_legacy_sdk_client_object():
    # Test legacy SDK returning object with embedding list / embedding.values
    mock_client = MagicMock()
    del mock_client.models
    
    mock_resp = MagicMock()
    mock_resp.embedding = [0.11, 0.22]
    mock_client.embed_content.return_value = mock_resp

    embedder = GeminiEmbedder(client=mock_client)
    result = embedder.embed_query("hello legacy object")

    assert result == [0.11, 0.22]


def test_gemini_embedder_legacy_sdk_client_object_values():
    mock_client = MagicMock()
    del mock_client.models
    
    mock_resp = MagicMock()
    mock_resp.embedding.values = [0.33, 0.44]
    mock_client.embed_content.return_value = mock_resp

    embedder = GeminiEmbedder(client=mock_client)
    result = embedder.embed_query("hello legacy object values")

    assert result == [0.33, 0.44]


def test_gemini_embedder_invalid_client():
    mock_client = MagicMock()
    # remove both models and embed_content to trigger ValueError
    del mock_client.models
    del mock_client.embed_content

    embedder = GeminiEmbedder(client=mock_client)
    with pytest.raises(ValueError, match="Provided client does not have embed_content method"):
        embedder.embed_query("hello")


# OpenAI Tests
def test_openai_embedder_missing_key():
    with patch.dict("os.environ", {}, clear=True):
        embedder = OpenAIEmbedder(api_key=None)
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            embedder.embed_query("test text")


@patch("urllib.request.urlopen")
def test_openai_embedder_rest_success(mock_urlopen):
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps({
        "data": [
            {
                "embedding": [0.01, -0.02, 0.03]
            }
        ]
    }).encode("utf-8")
    mock_urlopen.return_value.__enter__.return_value = mock_response

    embedder = OpenAIEmbedder(
        api_key="fake-openai-key",
        model_name="text-embedding-3-large",
        base_url="https://api.my-custom-proxy.com/v1"
    )
    result = embedder.embed_query("openai test")

    assert result == [0.01, -0.02, 0.03]

    args, kwargs = mock_urlopen.call_args
    req = args[0]
    assert req.full_url == "https://api.my-custom-proxy.com/v1/embeddings"
    assert req.headers["Authorization"] == "Bearer fake-openai-key"
    assert req.headers["Content-type"] == "application/json"  # urllib standardizes keys
    
    body_data = json.loads(req.data.decode("utf-8"))
    assert body_data == {
        "input": "openai test",
        "model": "text-embedding-3-large"
    }


@patch("urllib.request.urlopen")
def test_openai_embedder_rest_http_error(mock_urlopen):
    mock_fp = MagicMock()
    mock_fp.read.return_value = b"Unauthorized API Key"
    error = urllib.error.HTTPError(
        url="http://fake", code=401, msg="Unauthorized", hdrs=None, fp=mock_fp
    )
    mock_urlopen.side_effect = error

    embedder = OpenAIEmbedder(api_key="bad-key")
    with pytest.raises(RuntimeError, match="OpenAI API request failed with status 401: Unauthorized API Key"):
        embedder.embed_query("hello")


def test_openai_embedder_sdk_client():
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_emb_data = MagicMock()
    mock_emb_data.embedding = [-0.1, 0.5, 0.8]
    mock_resp.data = [mock_emb_data]
    mock_client.embeddings.create.return_value = mock_resp

    embedder = OpenAIEmbedder(client=mock_client, model_name="text-embedding-3-small")
    result = embedder.embed_query("openai sdk test")

    assert result == [-0.1, 0.5, 0.8]
    mock_client.embeddings.create.assert_called_once_with(
        input="openai sdk test", model="text-embedding-3-small"
    )


def test_openai_embedder_invalid_client():
    mock_client = MagicMock()
    del mock_client.embeddings

    embedder = OpenAIEmbedder(client=mock_client)
    with pytest.raises(ValueError, match="Provided client does not have embeddings.create method"):
        embedder.embed_query("hello")
