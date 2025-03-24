import unittest
import torch
from torch import nn
from cellvit.models.base.vision_transformer import VisionTransformer, PatchEmbed, Attention, Mlp

class TestVisionTransformer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.in_chans = 3
        self.embed_dim = 768
        self.num_classes = 10
        self.img_size = [224, 224]
        self.patch_size = 16
        self.num_heads = 12
        self.depth = 4

    def test_forward_pass(self):
        model = VisionTransformer(
            img_size=self.img_size,
            patch_size=self.patch_size,
            num_classes=self.num_classes,
            embed_dim=self.embed_dim
        )
        x = torch.randn(self.batch_size, self.in_chans, *self.img_size)
        with torch.no_grad():
            output = model(x)
        self.assertEqual(output.shape, (self.batch_size, self.embed_dim))

    def test_patch_embed(self):
        img_size = 224
        model = PatchEmbed(
            img_size=img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim
        )
        x = torch.randn(self.batch_size, self.in_chans, img_size, img_size)
        output = model(x)
        expected_num_patches = (img_size // self.patch_size) ** 2
        self.assertEqual(output.shape, (self.batch_size, expected_num_patches, self.embed_dim))

    def test_positional_embedding_interpolation(self):
        model = VisionTransformer(img_size=[224], patch_size=16)
        new_size = 256
        x = torch.randn(1, 3, new_size, new_size)
        tokens = model.prepare_tokens(x)
        expected_num_patches = (new_size // 16) ** 2 + 1  # +1 for cls_token
        self.assertEqual(tokens.shape[1], expected_num_patches)

    def test_class_token_presence(self):
        model = VisionTransformer()
        x = torch.randn(self.batch_size, 3, 224, 224)
        tokens = model.prepare_tokens(x)
        # Check class token is first and consistent across batch
        cls_tokens = tokens[:, 0, :]
        self.assertTrue(torch.allclose(cls_tokens[0], cls_tokens[1]))
        # Check total tokens
        num_patches = (224 // 16) ** 2
        self.assertEqual(tokens.shape[1], num_patches + 1)

    def test_attention_output_shapes(self):
        seq_length = 10
        model = Attention(dim=self.embed_dim, num_heads=self.num_heads)
        x = torch.randn(self.batch_size, seq_length, self.embed_dim)
        output, attn_weights = model(x)
        self.assertEqual(output.shape, (self.batch_size, seq_length, self.embed_dim))
        self.assertEqual(attn_weights.shape, (self.batch_size, self.num_heads, seq_length, seq_length))

    def test_mlp_output_shape(self):
        in_features = 768
        mlp = Mlp(in_features=in_features)
        x = torch.randn(self.batch_size, 10, in_features)
        output = mlp(x)
        self.assertEqual(output.shape, x.shape)

    def test_head_module_type(self):
        # Test classification head types
        model0 = VisionTransformer(num_classes=0)
        model10 = VisionTransformer(num_classes=10)
        self.assertIsInstance(model0.head, nn.Identity)
        self.assertIsInstance(model10.head, nn.Linear)

    def test_get_last_selfattention(self):
        model = VisionTransformer(depth=self.depth)
        x = torch.randn(1, 3, 224, 224)
        attn = model.get_last_selfattention(x)
        num_patches = (224 // 16) ** 2
        expected_shape = (1, 12, num_patches + 1, num_patches + 1)
        self.assertEqual(attn.shape, expected_shape)

    def test_get_intermediate_layers(self):
        model = VisionTransformer(depth=self.depth)
        x = torch.randn(1, 3, 224, 224)
        n = 2
        outputs = model.get_intermediate_layers(x, n=n)
        self.assertEqual(len(outputs), n)
        num_patches = (224 // 16) ** 2
        for output in outputs:
            self.assertEqual(output.shape, (1, num_patches + 1, self.embed_dim))

if __name__ == '__main__':
    unittest.main()