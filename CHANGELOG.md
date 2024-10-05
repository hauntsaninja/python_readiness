# Changelog

## [v1.1]

- Use bisection by default. This finds earlier supporting versions, but means there may also be
  intermediate versions satisfying the requirement without relevant support. You can use
  `--monotonic-support` to regain that guarantee, if useful, using the old linear search.
  Bisection is also significantly faster.
- Improve logic for handling yanked versions
- Use `requires-python` metadata to help determine support
- Add `--env` option to check a specific environment
- Show intermediate progress every 1 second
- Compress and version the cache, improve cache resiliency
- Add `--exclude-newer` option to exclude package versions newer than a given date
- Add a shorter timeout to HTTP requests
- Retry failed HTTP requests
- Fix support for Python 3.9
- Avoid checking installations originating from cwd
- Improve `--help` output

## [v1.0]

- Initial release
