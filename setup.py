from setuptools import setup, find_packages

setup(name='deepART',
      version='0.0.0',
      description='A library containing adaptive resonance theory neural networks',
      url='',
      author='Nicholas Cheng Xue Law, Chun Ping Lim',
      author_email='nlaw8@gatech.edu, lim.chunping@gmail.com',
      license='MIT',
      packages=find_packages(exclude=['*.tests']),
      install_requires=[
          'numpy==1.20.3',
          'joblib==1.0.1',
          'nltk==3.6.2',
          'pandas==1.2.4',
          'gensim==4.0.1',
          'spacy==3.0.6',
          "scikit-learn==0.24.2",
          "torch==1.8.1",
          "torchvision==0.9.1"
      ],
      include_package_data=True,
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False
      )