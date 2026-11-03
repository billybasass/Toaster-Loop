"""Non-production experiment code for toaster_loop work packages.

Everything under this directory is a RESEARCH SPIKE: never imported by the
``toaster_loop`` package, never packaged (setuptools only includes ``toaster_loop*``),
never wired into the CLI, worker, or dashboard. The deliverable of each spike
is its findings doc under ``docs/`` plus the measurements reproducible from
here. Heavy dependencies (torch) stay lazily imported inside functions,
matching the [training]-extra policy even though this tree is outside the
package's import-graph and license-hygiene scans by construction.
"""
