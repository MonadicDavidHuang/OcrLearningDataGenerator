import unittest

class TestHoge(unittest.TestCase):
    """test class of keisan.py
    """

    def test_hoge(self):
        """test method for tashizan
        """
        self.assertEqual(10, 10)

if __name__ == "__main__":
    unittest.main()