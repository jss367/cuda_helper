import pytest
from unittest.mock import patch
from main import get_cuda_version_unix

# Parametrized test for happy path scenarios
@pytest.mark.parametrize("mock_output, expected_version", [
    ("nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c)
    2005-2021 NVIDIA Corporation\nBuilt on Thu_Nov_18_09:44:06_Pacific_Standard_Time_2021\nCuda compilation tools, release 11.5, V11.5.119\nBuild cuda_11.5.r11.5/compiler.30672275_0", "11.5", {'id': 'CUDA_11.5'}),
    ("nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c)
    2005-2021 NVIDIA Corporation\nBuilt on Wed_Jan_15_20:08:12_Pacific_Standard_Time_2020\nCuda compilation tools, release 10.2, V10.2.89\nBuild cuda_10.2.r10.2/compiler.29558016_0", "10.2", {'id': 'CUDA_10.2'}),
], ids=["CUDA_11.5", "CUDA_10.2"])
def test_get_cuda_version_unix_happy_path(mock_output, expected_version):
    
    # Arrange
    with patch("main.get_version_via_command") as mock_get_version:
        mock_get_version.return_value = mock_output
        
    # Act
    result = get_cuda_version_unix()
    
    # Assert
    assert result == expected_version

# Parametrized test for edge cases
@pytest.mark.parametrize("mock_output, expected_version", [
    ("", None, {'id': 'Empty_output'}),
    ("nvcc: command not found", None, {'id': 'Command_not_found'}),
], ids=["Empty_output", "Command_not_found"])
def test_get_cuda_version_unix_edge_cases(mock_output, expected_version):
    
    # Arrange
    with patch("main.get_version_via_command") as mock_get_version:
        mock_get_version.return_value = mock_output
        
    # Act
    result = get_cuda_version_unix()
    
    # Assert
    assert result == expected_version

# Parametrized test for error cases
@pytest.mark.parametrize("mock_output, exception", [
    (Exception("Permission denied"), PermissionError, {'id': 'Permission_denied'}),
    (Exception("File not found"), FileNotFoundError, {'id': 'File_not_found'}),
], ids=["Permission_denied", "File_not_found"])
def test_get_cuda_version_unix_error_cases(mock_output, exception):
    
    # Arrange
    with patch("main.get_version_via_command", side_effect=mock_output):
        
    # Act & Assert
    with pytest.raises(exception):
        get_cuda_version_unix()
