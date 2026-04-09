BDI-Kit Documentation
=====================

**Version:** |version|


**BDI-Kit** is a toolkit designed to assist users in performing data harmonization (see our `GitHub repository <https://github.com/VIDA-NYU/bdi-kit>`__). It provides state-of-the-art tools to streamline the 
integration and transformation of disparate datasets, with a particular focus on biomedical data. BDI-Kit includes methods for tasks such as:

- Schema matching
- Value matching
- Data transformation to a target table or data model

BDI-Kit can be used in two complementary ways:

- 🐍 **Python API** — Programmatic data harmonization workflows
- 🤖 **AI Agent** — Conversational data harmonization using natural language

The following quick demo illustrates how BDI-Kit can be used through both the Python API and the AI agent:

.. image:: _static/images/demo_thumbnail.png
   :alt: Watch a demo of BDI-Kit
   :target: https://drive.google.com/file/d/1gMlZuocYrKFQYDZOphIyFj-nvjtx4ODR/view?usp=sharing
   :align: center

.. raw:: html

   <br><br>

For more details about the design and capabilities of BDI-Kit, see our papers:

- `BDI-Kit: An AI-Powered Toolkit for Biomedical Data Harmonization <https://www.cell.com/action/showPdf?pii=S2666-3899%2825%2900318-6>`__  (preferred citation)
- `BDI-Kit Demo: A Toolkit for Programmable and Conversational Data Harmonization <https://arxiv.org/pdf/2604.06405>`__


.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   installation
   quick-start
   getting-started
   examples
   contributing

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   api
   schema-matching
   value-matching
