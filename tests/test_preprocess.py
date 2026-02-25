import unittest
from PIL import Image
import numpy as np
import os
from src.preprocess import load_and_preprocess

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        self.img_path = "test_image.jpg"
        self.img = Image.fromarray(np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8))
        self.img.save(self.img_path)

    def tearDown(self):
        if os.path.exists(self.img_path):
            os.remove(self.img_path)

    def test_resize_dimensions(self):
        target_size = 224
        processed = load_and_preprocess(self.img_path, target_size=target_size, pad_to_square=False)
        self.assertIsNotNone(processed)
        w, h = processed.size
        self.assertTrue(w == target_size or h == target_size)
        self.assertTrue(w <= target_size and h <= target_size)

    def test_pad_to_square(self):
        target_size = 224
        processed = load_and_preprocess(self.img_path, target_size=target_size, pad_to_square=True)
        self.assertIsNotNone(processed)
        w, h = processed.size
        self.assertEqual(w, target_size)
        self.assertEqual(h, target_size)

if __name__ == '__main__':
    unittest.main()
