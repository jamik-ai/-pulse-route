"""Sphinx: документация кода «Пульс маршрута» (собирается в Docker-образе, отдаётся backend'ом по /code-docs/)."""
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
project = "Пульс маршрута"
author = "Команда venik"
copyright = "2026, команда venik"
language = "ru"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "sphinx.ext.autosectionlabel"]
autosectionlabel_prefix_document = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {"members": True, "show-inheritance": False}
autodoc_mock_imports = ["catboost", "lightgbm", "sklearn"]
napoleon_google_docstring = True

html_theme = "furo"
html_title = "Пульс маршрута"
html_short_title = "Пульс маршрута"
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = False
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#0A6CFF",
        "color-brand-content": "#0A6CFF",
        "color-admonition-background": "#F5F7FB",
        "font-stack": "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Segoe UI', Roboto, sans-serif",
        "font-stack--monospace": "'SF Mono', Menlo, Consolas, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4DA3FF",
        "color-brand-content": "#4DA3FF",
    },
    "footer_icons": [],
}
