#!/bin/zsh

rm -rf .deploy_git

/usr/local/bin/hexo cl
/usr/local/bin/hexo d -g