"""coding: utf-8"""

import unittest


class TestHoge(unittest.TestCase):
    """Test class of Hoge.
    """

    def test(self):
        """Test.
        """
        self.assertEqual(10, 10)


if __name__ == "__main__":
    unittest.main()
