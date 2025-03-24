import unittest
from unittest.mock import patch, MagicMock
import io
import sys
import ray
from cellvit.utils.check_ray import log_message, test_import, main


class TestCheckRay(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Initialize Ray before all tests."""
        ray.init(ignore_reinit_error=True)
    
    @classmethod
    def tearDownClass(cls):
        """Shut down Ray after all tests."""
        ray.shutdown()
    
    def test_log_message_debug(self):
        """Test of the log_message function with DEBUG level."""
        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
            log_message("Test message", level="DEBUG")
            output = fake_stdout.getvalue()
            self.assertIn("Test message", output)
            self.assertIn("[DEBUG]", output)
    
    def test_log_message_info(self):
        """Test of the log_message function with INFO level."""
        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
            log_message("Level test message", level="INFO")
            output = fake_stdout.getvalue()
            self.assertIn("[INFO]", output)
            self.assertIn("Level test message", output)
    
    def test_log_message_warning(self):
        """Test of the log_message function with WARNING level."""
        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
            log_message("Level test message", level="WARNING")
            output = fake_stdout.getvalue()
            self.assertIn("[WARNING]", output)
            self.assertIn("Level test message", output)
    
    def test_log_message_error(self):
        """Test of the log_message function with ERROR level."""
        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
            log_message("Level test message", level="ERROR")
            output = fake_stdout.getvalue()
            self.assertIn("[ERROR]", output)
            self.assertIn("Level test message", output)
    
    def test_test_import(self):
        """Test of the test_import function."""
        result = ray.get(test_import.remote())
        self.assertIn(result, ["Success", "CuPy is not available."])
    
    @patch('sys.exit')
    def test_main_function(self, mock_exit):
        """Test of the main function."""
        status_code = main()
        self.assertIn(status_code, [0, 1])

    @patch('sys.exit')
    def test_main_function_success(self, mock_exit):
        """Test of the main function with successful Ray initialization."""
        try:
            self.tearDownClass()
            self.setUpClass()
        except:
            pass
        
        with patch('ray.init') as mock_init, \
            patch('ray.get') as mock_get, \
            patch('ray.shutdown') as mock_shutdown:
            # Erfolgreichen Durchlauf simulieren
            mock_get.return_value = "Success"
            
            status_code = main()
            
            # Prüfen, dass Ray korrekt initialisiert und beendet wurde
            mock_shutdown.assert_called()
            
            # Status Code sollte 1 sein (Erfolg)
            self.assertEqual(status_code, 1)

    @patch('sys.exit')
    def test_main_function_ray_init_error(self, mock_exit):
        """Test of the main function with Ray initialization error."""
        with patch('ray.init') as mock_init:
            # Ray-Initialisierungsfehler simulieren
            mock_init.side_effect = Exception("Ray init error")
            
            status_code = main()
            
            # Status Code sollte 0 sein (Fehler)
            self.assertEqual(status_code, 0)
            
            # Prüfen, dass keine weiteren Ray-Operationen versucht wurden
            mock_init.assert_called()

    @patch('sys.exit')
    def test_main_function_worker_import_fail(self, mock_exit):
        """Test of the main function with worker import error."""
        with patch('ray.init'), \
            patch('ray.get') as mock_get, \
            patch('ray.shutdown'):
            # Worker-Importfehler simulieren
            mock_get.return_value = "Failed to import cellvit in worker: Some error"
            
            status_code = main()
            
            # Status Code sollte 0 sein (Fehler)
            self.assertEqual(status_code, 0)

    @patch('sys.exit')
    def test_main_function_worker_communication_error(self, mock_exit):
        """Test of the main function with worker communication error."""
        with patch('ray.init'), \
            patch('ray.get') as mock_get, \
            patch('ray.shutdown'):
            # Ray Worker Kommunikationsfehler simulieren
            mock_get.side_effect = Exception("Worker communication error")
            
            status_code = main()
            
            # Status Code sollte 0 sein (Fehler)
            self.assertEqual(status_code, 0)

    @patch('sys.exit')
    def test_main_function_with_cupy(self, mock_exit):
        """Test of the main function with available CuPy."""
        try:
            self.tearDownClass()
            self.setUpClass()
        except:
            pass
        with patch('ray.init'), \
            patch('ray.get') as mock_get, \
            patch('ray.shutdown'), \
            patch.dict('sys.modules', {'cupy': MagicMock()}):  # CuPy als verfügbar simulieren
            #  spatch.dict('sys.modules', {'cupy': MagicMock()}):  # CuPy als verfügbar simulieren

            mock_get.return_value = "Success"
            
            # Ausgabe erfassen und prüfen
            with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
                status_code = main()
                output = fake_stdout.getvalue()
                
                self.assertEqual(status_code, 1)
                self.assertIn("CuPy is available", output)

    @patch('sys.exit')
    def test_main_function_without_cupy(self, mock_exit):
        """Test of the main function without available CuPy."""
        try:
            self.tearDownClass()
            self.setUpClass()
        except:
            pass
        with patch('ray.init'), \
            patch('ray.get') as mock_get, \
            patch('ray.shutdown'), \
            patch.dict('sys.modules', {}):  # CuPy als nicht verfügbar simulieren
            
            # ImportError für cupy simulieren
            import builtins
            original_import = builtins.__import__
            
            def mock_import(name, *args, **kwargs):
                if name == 'cupy':
                    raise ImportError("No module named 'cupy'")
                return original_import(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=mock_import):
                mock_get.return_value = "CuPy is not available."
                
                # Ausgabe erfassen und prüfen
                with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
                    status_code = main()
                    output = fake_stdout.getvalue()
                    
                    self.assertEqual(status_code, 1)
                    self.assertIn("CuPy is not available", output)
                    self.assertIn("Successfully imported cellvit (numpy)", output)

if __name__ == '__main__':
    unittest.main()