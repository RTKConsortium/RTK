#!/bin/bash

git add .
read COMM
git commit -m "$COMM"
git push 
