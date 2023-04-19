import unittest
from pathlib import Path

if __name__ == '__main__':
    testsuite = unittest.TestLoader().discover(Path(__file__).parent)
    unittest.TextTestRunner(verbosity=2).run(testsuite)
