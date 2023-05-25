# Matcher

[![Telegram](https://img.shields.io/badge/chat-on%20Telegram-2ba2d9.svg)](https://t.me/lamamatcher)

<a name="installation"></a>
# Installation
To install LAMA framework with Matcher on your machine, execute following commands:  

Download last version of wheel from repository. [Alpha](https://df-bitbucket.sbrf.ru/projects/MATCHER/repos/lama-matcher/browse) [Sigma](https://stash.delta.sbrf.ru/projects/MATCHER/repos/lama-matcher/browse) 
```bash

# Install base functionality:

pip install -U lightautoml-<vesion>.whl


# Or you can build last version using poetry

# Sigma
git clone https://stash.delta.sbrf.ru/scm/finds/matcher.git mathcer

# Alpha 
git clone https://df-bitbucket.delta.sbrf.ru/scm/finds/matcher.git matcher

cd matcher
poetry build 
pip install dist/lightautoml-<version>.whl
```

