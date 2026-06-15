"""E2E smoke test: navigation between pages loads without error.

Navigation is exercised through the top navbar. These tests assert on static UI
(the navbar, the selection sidebar, the Binding workspace heading) that renders
immediately, so they do not depend on the background data initialization /
materialization completing first.
"""

from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture

app = create_app_fixture("../../tfbpshiny/app.py")


def test_home_loads(page: Page, app):
    page.goto(app.url)
    expect(page.locator(".navbar")).to_be_visible()


def test_navigate_to_selection(page: Page, app):
    page.goto(app.url)
    # Navbar tabs are bslib nav-links carrying the panel name in data-value
    # (role="tab", not "link"). data-value is unique to the top navbar tab.
    page.locator('a.nav-link[data-value="Dataset selection"]').click()
    expect(page.locator(".selection-sidebar")).to_be_visible()


def test_navigate_to_binding(page: Page, app):
    page.goto(app.url)
    # "Binding" is exact: distinct from "Binding/Perturbation Comparisons".
    page.locator('a.nav-link[data-value="Binding"]').click()
    expect(page.get_by_role("heading", name="Binding Correlation")).to_be_visible()
