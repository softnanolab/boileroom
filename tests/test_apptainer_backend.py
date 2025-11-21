"""Tests for Apptainer backend."""
# Jakub Note: these are vibe-coded and should likely not be used, but let's keep them if relevant and for context.

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from boileroom.backend.apptainer import (
    _extract_device_number,
    _get_cached_sif_path,
    _get_image_name,
    _is_image_cached,
    _is_tool_available,
    ApptainerBackend,
)


class TestImageNameResolution:
    """Test image name and URI generation."""

    def test_get_image_name_default(self):
        """Test generating image name with default registry."""
        result = _get_image_name("chai")
        assert result == "docker://docker.io/jakublala/boileroom-chai:latest"

    def test_get_image_name_custom_registry(self):
        """Test generating image name with custom registry."""
        result = _get_image_name("boltz", registry="ghcr.io/myorg")
        assert result == "docker://ghcr.io/myorg/boileroom-boltz:latest"


class TestCachePathGeneration:
    """Test cache path generation from image URIs."""

    def test_get_cached_sif_path(self, tmp_path):
        """Test generating cache path for .sif file."""
        image_uri = "docker://docker.io/jakublala/boileroom-chai1:latest"
        cache_dir = tmp_path / "cache"
        result = _get_cached_sif_path(image_uri, cache_dir)
        
        expected = cache_dir / "images" / "jakublala-boileroom-chai1_latest.sif"
        assert result == expected

    def test_get_cached_sif_path_with_tag(self, tmp_path):
        """Test cache path generation with different tags."""
        image_uri = "docker://docker.io/jakublala/boileroom-esm:v1.0.0"
        cache_dir = tmp_path / "cache"
        result = _get_cached_sif_path(image_uri, cache_dir)
        
        expected = cache_dir / "images" / "jakublala-boileroom-esm_v1.0.0.sif"
        assert result == expected


class TestImageCaching:
    """Test image caching logic."""

    def test_is_image_cached_exists(self, tmp_path):
        """Test checking if cached image exists."""
        sif_path = tmp_path / "image.sif"
        sif_path.write_bytes(b"fake image data")
        assert _is_image_cached(sif_path) is True

    def test_is_image_cached_not_exists(self, tmp_path):
        """Test checking if cached image doesn't exist."""
        sif_path = tmp_path / "nonexistent.sif"
        assert _is_image_cached(sif_path) is False

    def test_is_image_cached_empty_file(self, tmp_path):
        """Test that empty files are not considered valid."""
        sif_path = tmp_path / "empty.sif"
        sif_path.touch()
        assert _is_image_cached(sif_path) is False


class TestDeviceNumberExtraction:
    """Test device number extraction."""

    def test_extract_device_number_cuda(self):
        """Test extracting device number from CUDA device string."""
        assert _extract_device_number("cuda:0") == "0"
        assert _extract_device_number("cuda:1") == "1"
        assert _extract_device_number("cuda:2") == "2"

    def test_extract_device_number_cpu(self):
        """Test extracting device number from CPU device string."""
        assert _extract_device_number("cpu") is None

    def test_extract_device_number_invalid(self):
        """Test extracting device number from invalid device string."""
        assert _extract_device_number("invalid") is None


class TestToolAvailability:
    """Test tool availability checking."""

    @patch("shutil.which")
    def test_is_tool_available_true(self, mock_which):
        """Test checking if tool is available."""
        mock_which.return_value = "/usr/bin/apptainer"
        assert _is_tool_available("apptainer") is True
        mock_which.assert_called_once_with("apptainer")

    @patch("shutil.which")
    def test_is_tool_available_false(self, mock_which):
        """Test checking if tool is not available."""
        mock_which.return_value = None
        assert _is_tool_available("apptainer") is False
        mock_which.assert_called_once_with("apptainer")


class TestApptainerBackendInit:
    """Test ApptainerBackend initialization."""

    @patch("shutil.which")
    def test_init_apptainer_not_installed(self, mock_which):
        """Test that ValueError is raised when apptainer is not installed."""
        mock_which.return_value = None
        
        with pytest.raises(ValueError, match="To use the ApptainerBackend"):
            ApptainerBackend(
                core_class_path="boileroom.models.esm.core.ESM2Core",
                image_uri="docker://docker.io/jakublala/boileroom-esm:latest",
            )

    @patch("shutil.which")
    @patch("boileroom.backend.apptainer.ensure_cache_dir")
    def test_init_apptainer_installed(self, mock_cache_dir, mock_which, tmp_path):
        """Test successful initialization when apptainer is installed."""
        mock_which.return_value = "/usr/bin/apptainer"
        mock_cache_dir.return_value = tmp_path
        
        backend = ApptainerBackend(
            core_class_path="boileroom.models.esm.core.ESM2Core",
            image_uri="docker://docker.io/jakublala/boileroom-esm:latest",
        )
        
        assert backend._core_class_path == "boileroom.models.esm.core.ESM2Core"
        assert backend._image_uri == "docker://docker.io/jakublala/boileroom-esm:latest"
        assert backend._device == "cuda:0"  # default
        assert backend._sif_path.parent == tmp_path / "images"

    @patch("shutil.which")
    @patch("boileroom.backend.apptainer.ensure_cache_dir")
    def test_init_custom_cache_dir(self, mock_cache_dir, mock_which, tmp_path):
        """Test initialization with custom cache directory."""
        mock_which.return_value = "/usr/bin/apptainer"
        custom_cache = tmp_path / "custom"
        
        backend = ApptainerBackend(
            core_class_path="boileroom.models.esm.core.ESM2Core",
            image_uri="docker://docker.io/jakublala/boileroom-esm:latest",
            cache_dir=custom_cache,
        )
        
        assert backend._cache_dir == custom_cache
        assert backend._sif_path.parent == custom_cache / "images"

