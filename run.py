import sys
print(sys.executable)


import os
print(os.path.abspath("."))

with open("README.md", "w") as f:
  f.write("test script")
