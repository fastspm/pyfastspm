============
Contributing
============
Contributions are welcome, and they are greatly appreciated! Every little bit helps, 
and credit will always be given.

Types of Contributions
======================

Report Bugs
-----------

Report bugs at https://gitlab.com/fastspm/pyfastspm/-/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
--------

Look through the Bitbucket issues for bugs. Anything tagged with "bug"
is open to whoever wants to fix it.

Implement Features
------------------

Look through the GitLab issues for features. Anything tagged with "feature"
is open to whoever wants to implement it.

Write Documentation
-------------------

FAST movie processor could always use more documentation, whether as part of the
official FAST movie processor docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
---------------

The best way to send feedback is to file an issue at https://gitlab.com/fastspm/pyfastspm/-/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions are welcome :)

Get Started!
============

Ready to contribute? Here's how to set up ``pyfastspm`` for
local development.

Installing the package in development mode
------------------------------------------
Follow these instructions if you want to actively develop the  ``pyfastspm`` package. 
It is strongly recommended to use the ``mambaforge`` distribution (see :doc:`/installation`)

1. clone the ``pyfastspm`` git repository on your system;
2. in a terminal, ``cd`` into the cloned repository
3. create the ``conda`` enviroment described inside the ``tests/environment-dev.yml``, using the following command:
   
   .. code-block:: bash

       mamba create -f tests/environment-dev.yml

4. activate the newly created enviroment:

   .. code-block:: bash

       conda activate pyfastspm-dev

5. to use and test the package making it available in the whole environment, use the 
following command issued from the terminal:

   .. code-block:: bash

       pip install -e . --no-deps

This installs the package in development mode, so that every modification to the source is 
immediately available to be imported by any package living in in the same environment.

Working on the code
-------------------
The workflow for contribute to the project code are generally:

1. create a local branch for development::

    $ git checkout -b name-of-your-bugfix-or-feature

2. if necessary, update the test suite to include the functionality you have just developed/fixed.

2. When you're done making changes, check that your changes pass the unit tests with ``pytest``::

    $ pytest -vx
    $ pytest -vx --nbmake examples/pyfastspm_converter.ipynb

5. Update the ``CHANGELOG.md`` file with a detailed description of your changes

5. Commit your changes and push your branch to GitLab::

    $ git add .
    $ git commit -m "Your short description of your changes."
    $ git push -u origin name-of-your-bugfix-or-feature

6. Submit a pull request through the GitLab website.
