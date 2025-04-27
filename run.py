#!/usr/bin/env python
import os
import sys

# Add the app directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))

from app import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 