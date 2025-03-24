# -*- coding: utf-8 -*-
# Test linear classifier for CellViT
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import unittest
import torch
from cellvit.models.classifier.linear_classifier import LinearClassifier


class TestLinearClassifier(unittest.TestCase):
    def test_linear_classifier_forward(self):
        """Test case for forward pass of LinearClassifier"""
        # Test case for forward pass
        embed_dim = 128
        hidden_dim = 64
        num_classes = 10
        drop_rate = 0.1

        model = LinearClassifier(embed_dim, hidden_dim, num_classes, drop_rate)
        model.eval()  # Set model to evaluation mode

        # Create a dummy input tensor with batch size 4 and embedding dimension 128
        input_tensor = torch.randn(4, embed_dim)

        # Perform forward pass
        output = model(input_tensor)

        # Check output shape
        self.assertEqual(
            output.shape,
            (4, num_classes),
            f"Expected output shape (4, {num_classes}), but got {output.shape}",
        )

    def test_linear_classifier_dropout(self):
        """Test case to check if dropout is applied"""
        # Test case to check if dropout is applied
        embed_dim = 128
        hidden_dim = 64
        num_classes = 10
        drop_rate = 0.5

        model = LinearClassifier(embed_dim, hidden_dim, num_classes, drop_rate)
        model.train()  # Set model to training mode

        # Create a dummy input tensor
        input_tensor = torch.randn(4, embed_dim)

        # Perform forward pass
        output = model(input_tensor)

        # Check if dropout is applied (output should not be identical to input)
        self.assertFalse(
            torch.equal(output, model.fc2(model.activation(model.fc1(input_tensor))))
        )

    def test_linear_classifier_no_dropout(self):
        """Test case to check behavior when dropout is 0"""
        # Test case to check behavior when dropout is 0
        embed_dim = 128
        hidden_dim = 64
        num_classes = 10
        drop_rate = 0.0

        model = LinearClassifier(embed_dim, hidden_dim, num_classes, drop_rate)
        model.train()  # Set model to training mode

        # Create a dummy input tensor
        input_tensor = torch.randn(4, embed_dim)

        # Perform forward pass
        output = model(input_tensor)

        # Check if output is consistent (dropout should not alter the output)
        self.assertTrue(
            torch.equal(output, model.fc2(model.activation(model.fc1(input_tensor))))
        )

    def test_linear_classifier_initialization(self):
        """Test case to check if model initializes correctly"""
        # Test case to check if model initializes correctly
        embed_dim = 128
        hidden_dim = 64
        num_classes = 10
        drop_rate = 0.1

        model = LinearClassifier(embed_dim, hidden_dim, num_classes, drop_rate)

        # Check if layers are initialized with correct dimensions
        self.assertEqual(model.fc1.in_features, embed_dim)
        self.assertEqual(model.fc1.out_features, hidden_dim)
        self.assertEqual(model.fc2.in_features, hidden_dim)
        self.assertEqual(model.fc2.out_features, num_classes)

    def test_linear_classifier_activation(self):
        """Test case to check if activation function is applied"""
        # Test case to check if activation function is applied
        embed_dim = 128
        hidden_dim = 64
        num_classes = 10
        drop_rate = 0.1

        model = LinearClassifier(embed_dim, hidden_dim, num_classes, drop_rate)
        model.eval()  # Set model to evaluation mode

        # Create a dummy input tensor
        input_tensor = torch.randn(4, embed_dim)

        # Perform forward pass
        intermediate_output = model.activation(model.fc1(input_tensor))
        output = model(input_tensor)

        # Check if activation is applied correctly
        self.assertTrue(
            torch.equal(intermediate_output, model.activation(model.fc1(input_tensor)))
        )


if __name__ == "__main__":
    unittest.main()
