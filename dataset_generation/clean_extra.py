import os
import sys

for f in os.listdir('.'):
	if (f.count('-') == 1):
		t = os.path.join(sys.path[0],f)
		os.remove(t)
