import unittest
from pathlib import Path
from unittest.mock import patch
import io
import shutil 
from cellvit.inference.cli import InferenceWSIParser


class TestInferenceWSIParser(unittest.TestCase):
    def setUp(self):
        # Beispiel-Konfiguration für Tests
        self.example_yaml_config = """
        model: SAM
        nuclei_taxonomy: pannuke
        inference:
          gpu: 0
          enforce_amp: true
          batch_size: 8
        output_format:
          outdir: output
          geojson: true
          graph: false
          compression: true
        process_wsi:
          wsi_path: tests/test_data/sample_wsi.svs
          wsi_mpp: 0.25
          wsi_magnification: 20
        system:
          cpu_count: 4
          ray_worker: 2
          ray_remote_cpus: 2
          memory: 8192
        debug: false
        """
        
        # Temporäre Verzeichnisse und Dateien erstellen
        self.temp_dir = Path("tests/test_data")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.temp_wsi_file = self.temp_dir / "sample_wsi.svs"
        self.temp_wsi_file.touch()
        
        # Beispiel-Argumente für CLI-Tests
        self.wsi_cli_args = {
            'config': None,
            'model': 'SAM',
            'nuclei_taxonomy': 'pannuke',
            'gpu': 0,
            'enforce_amp': True,
            'batch_size': 8,
            'outdir': 'output',
            'geojson': True,
            'graph': False,
            'compression': True,
            'mode': 'process_wsi',
            'wsi_path': 'tests/test_data/sample_wsi.svs',
            'wsi_mpp': 0.25,
            'wsi_magnification': 20,
            'cpu_count': 4,
            'ray_worker': 2,
            'ray_remote_cpus': 2,
            'memory': 8192,
            'debug': False
        }
        
        self.dataset_cli_args = {
            'config': None,
            'model': 'SAM',
            'nuclei_taxonomy': 'pannuke',
            'gpu': 0,
            'enforce_amp': True,
            'batch_size': 8,
            'outdir': 'output',
            'geojson': True,
            'graph': False,
            'compression': True,
            'mode': 'process_dataset',
            'wsi_folder': 'tests/test_data',
            'wsi_filelist': None,
            'wsi_extension': 'svs',
            'wsi_mpp': 0.25,
            'wsi_magnification': 20,
            'cpu_count': 4,
            'ray_worker': 2,
            'ray_remote_cpus': 2,
            'memory': 8192,
            'debug': False
        }

    def tearDown(self):
        # Temporäre Dateien aufräumen
        if self.temp_wsi_file.exists():
            self.temp_wsi_file.unlink()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_parser_initialization(self):
        """Test for parser initialization."""
        parser = InferenceWSIParser()
        self.assertIsNotNone(parser.parser, "Parser should be initialized")

    @patch('argparse.ArgumentParser.parse_args')
    @patch('torch.cuda.device_count')
    def test_parse_wsi_arguments(self, mock_device_count, mock_parse_args):
        """Test for parsing WSI process arguments."""
        mock_device_count.return_value = 1
        
        # Argparse.Namespace statt MagicMock verwenden
        from argparse import Namespace
        namespace_args = Namespace(**self.wsi_cli_args)
        mock_parse_args.return_value = namespace_args
        
        # Parser initialisieren und Argumente parsen
        parser = InferenceWSIParser()
        config = parser.parse_arguments()
        
        # Überprüfen der erzeugten Konfiguration
        self.assertEqual(config.model, "SAM", "Model should be SAM")
        self.assertEqual(config.nuclei_taxonomy, "pannuke", "Nuclei taxonomy should be pannuke")
        self.assertEqual(config.gpu, 0, "GPU should be 0")
        self.assertTrue(config.enforce_amp, "Enforce AMP should be True")
        self.assertEqual(config.batch_size, 8, "Batch size should be 8")
        self.assertEqual(config.outdir, Path("output"), "Output directory should be 'output'")
        self.assertTrue(config.geojson, "GeoJSON should be True")
        self.assertFalse(config.graph, "Graph should be False")
        self.assertTrue(config.compression, "Compression should be True")
        self.assertEqual(config.command, "process_wsi", "Command should be 'process_wsi'")
        self.assertEqual(config.wsi_path, Path("tests/test_data/sample_wsi.svs"), "WSI path should be 'tests/test_data/sample_wsi.svs'")
        self.assertEqual(config.wsi_mpp, 0.25, "WSI MPP should be 0.25")
        self.assertEqual(config.wsi_magnification, 20, "WSI magnification should be 20")

    @patch('argparse.ArgumentParser.parse_args')
    @patch('torch.cuda.device_count')
    def test_parse_dataset_arguments(self, mock_device_count, mock_parse_args):
        """Test for parsing dataset process arguments."""
        mock_device_count.return_value = 1
        
        # Argparse-Ergebnis mocken
        from argparse import Namespace
        namespace_args = Namespace(**self.dataset_cli_args)
        mock_parse_args.return_value = namespace_args
        
        # Parser initialisieren und Argumente parsen
        parser = InferenceWSIParser()
        config = parser.parse_arguments()
        
        # Überprüfen der erzeugten Konfiguration
        self.assertEqual(config.model, "SAM", "Model should be SAM")
        self.assertEqual(config.command, "process_dataset", "Command should be 'process_dataset'")
        self.assertEqual(config.wsi_folder, Path("tests/test_data"), "WSI folder should be 'tests/test_data'")
        self.assertEqual(config.wsi_extension, "svs", "WSI extension should be 'svs'")
        self.assertIsNone(config.wsi_path, "WSI path should be None")


    def test_transform_cli_to_yaml(self):
        """Test for transforming CLI arguments to YAML structure."""
        parser = InferenceWSIParser()
        yaml_config = parser._transform_cli_to_yaml_structure(self.wsi_cli_args)
        
        # Überprüfen der erzeugten YAML-Struktur
        self.assertEqual(yaml_config["model"], "SAM", "Model should be SAM")
        self.assertEqual(yaml_config["nuclei_taxonomy"], "pannuke", "Nuclei taxonomy should be pannuke")
        self.assertEqual(yaml_config["inference"]["gpu"], 0, "GPU should be 0")
        self.assertTrue(yaml_config["inference"]["enforce_amp"], "Enforce AMP should be True")
        self.assertEqual(yaml_config["inference"]["batch_size"], 8, "Batch size should be 8")
        self.assertEqual(yaml_config["output_format"]["outdir"], "output", "Output directory should be 'output'")
        self.assertTrue(yaml_config["output_format"]["geojson"], "GeoJSON should be True")
        self.assertFalse(yaml_config["output_format"]["graph"], "Graph should be False")
        self.assertTrue(yaml_config["output_format"]["compression"], "Compression should be True")
        self.assertEqual(yaml_config["process_wsi"]["wsi_path"], "tests/test_data/sample_wsi.svs", "WSI path should be 'tests/test_data/sample_wsi.svs'")
        self.assertEqual(yaml_config["process_wsi"]["wsi_mpp"], 0.25, "WSI MPP should be 0.25")
        self.assertEqual(yaml_config["process_wsi"]["wsi_magnification"], 20, "WSI magnification should be 20")
    

    @patch('argparse.ArgumentParser.parse_args')
    @patch('torch.cuda.device_count')
    def test_cli_help_message(self, mock_device_count, mock_parse_args):
        """Test for displaying help message."""
        mock_device_count.return_value = 1
        
        # Simuliere, dass parse_args eine SystemExit auslöst
        mock_parse_args.side_effect = SystemExit()
        
        # Erstelle einen Parser
        parser = InferenceWSIParser()
        
        # Fange stdout auf und prüfe, ob die Hilfe-Nachricht ausgegeben wird
        with patch('sys.stdout', new=io.StringIO()) as fake_stdout:
            with self.assertRaises(SystemExit):
                # parse_arguments aufrufen, was intern parse_args aufruft
                parser.parse_arguments()
                
            # Da wir SystemExit mocken, wird kein Output erzeugt
            # Hier sollten wir stattdessen die print_help-Methode direkt aufrufen
            parser.parser.print_help()
            output = fake_stdout.getvalue()
            self.assertIn("CellViT++", output, "Help message should contain 'CellViT++'")