"""PYORPS GUI callbacks: thin wiring between UI events and the services."""
from __future__ import annotations


def register_all(app, state) -> None:
    """Register every callback module against the app + shared state."""
    from . import (attrs, browse, constrained, costmodel, data, edit, groups,
                   guide, interaction, io, layers, notifications, raster,
                   routes, theme)

    notifications.register(app, state)
    layers.register(app, state)
    attrs.register(app, state)
    data.register(app, state)
    raster.register(app, state)
    costmodel.register(app, state)
    interaction.register(app, state)
    routes.register(app, state)
    edit.register(app, state)
    groups.register(app, state)
    io.register(app, state)
    constrained.register(app, state)
    browse.register(app, state)
    guide.register(app, state)
    theme.register(app, state)
