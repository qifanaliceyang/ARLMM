MAJOR = 2
MINOR = 0
PATCH = 0
PRE_RELEASE = 'rc0'

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)


__shortversion__ = '.'.join(map(str, VERSION[:3]))
__version__ = '.'.join(map(str, VERSION[:3])) + ''.join(VERSION[3:])

__package_name__ = 'ARLMM_toolkit'
__contact_names__ = 'Qifan Yang'
__contact_emails__ = 'Qifan.Yang@bcm.edu'
__homepage__ = 'https://docs.google.com/document/d/1wNjiqPMkySHePyFwfzNUbk9dozctEk1rkr2XCKgufjU/edit?usp=sharing'
__repository_url__ = 'https://github.com/nvidia/ARLMM'
__download_url__ = 'https://github.com/qifanaliceyang/ARLMM/releases'
__description__ = 'ARLMM - Autoregressive Linear Mixed Models for genetic associations with temporally or spatially correlated phenotypes'
__license__ = 'MIT'
__keywords__ = 'deep learning, machine learning, gpu, NLP, NeMo, nvidia, pytorch, torch, tts, speech, language'
