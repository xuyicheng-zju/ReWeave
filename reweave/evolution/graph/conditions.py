# Modified for the ReWeave namespace.
"""Readable finite-domain coordinates for one canonical semantic graph.

Definitions form a shared vocabulary. Object names are display information;
physical qualifications and explicitly scoped task requirements are facts.
Unknown is an absent constraint, never an additional physical value.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConditionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FluentDefinition(ConditionModel):
    definition: str = Field(
        min_length=1,
        description="Exact meaning of one state variable, including its temporal or task-requirement scope.",
    )
    arguments: tuple[str, ...] = Field(
        description="Meanings of the ordered entity arguments; reversing arguments changes the grounded variable. A phase-dependent requirement must include its phase/context as an argument, not hide it in 'current stage'. Physical properties keep their actual physical arguments."
    )
    values: dict[str, str] = Field(
        min_length=2,
        description="Mutually exclusive values of this one property, with precise meanings. The domain is immutable; unknown is absence of a constraint.",
    )

    @model_validator(mode="after")
    def validate_definition(self):
        if any(
            not text.strip()
            for text in (
                self.definition,
                *self.arguments,
                *self.values,
                *self.values.values(),
            )
        ):
            raise ValueError("fluent definitions and value names must be nonempty")
        return self


class StateCondition(ConditionModel):
    fluent: str = Field(
        min_length=1,
        description="Definition ID in this graph's shared finite-domain catalog.",
    )
    arguments: tuple[str, ...] = Field(
        description="Graph-wide entity IDs in the definition's argument order; the same IDs retain identity across all states."
    )
    values: tuple[str, ...] = Field(
        min_length=1,
        description="Allowed values, a nonempty proper subset of the fluent domain. Repeated constraints on one grounded fluent are intersected.",
    )


class GraphEntity(ConditionModel):
    description: str = Field(
        min_length=1,
        description="Human-readable referent, such as a physical object or a declared task phase. Display names do not constrain cross-task substitution.",
    )
    conditions: tuple[StateCondition, ...] = Field(
        default=(),
        description="Explicit qualifications or task requirements fixed throughout this graph. Stage-specific requirements belong to the node or interface. Names such as targetA are not constraints.",
    )


class InterfaceConditions(ConditionModel):
    entry: tuple[StateCondition, ...] = Field(
        default=(),
        description="Conditions at this particular transition's entrance; alternative edges retain separate interfaces.",
    )
    exit: tuple[StateCondition, ...] = Field(
        default=(),
        description="Conditions at this transition's intended exit. Witness outcomes remain separate evidence and may be negative.",
    )


def normalize_conditions(conditions, catalog, entities):
    """A conjunction has exactly one intersection per grounded variable."""
    intersections = {}
    for condition in conditions:
        if condition.fluent not in catalog:
            raise ValueError("condition references an undeclared fluent")
        definition = catalog[condition.fluent]
        if len(condition.arguments) != len(definition.arguments):
            raise ValueError("condition argument arity differs from its definition")
        if not set(condition.arguments) <= entities:
            raise ValueError("condition references an undeclared entity")
        domain, values = set(definition.values), set(condition.values)
        if not values or not values < domain:
            raise ValueError("condition values must be a nonempty proper domain subset")
        key = (condition.fluent, condition.arguments)
        intersections[key] = intersections.get(key, domain) & values
        if not intersections[key]:
            raise ValueError("contradictory constraints in one condition context")
    return tuple(
        StateCondition(fluent=fluent, arguments=arguments, values=tuple(sorted(values)))
        for (fluent, arguments), values in sorted(intersections.items())
    )


class GraphSemantics(ConditionModel):
    """The matching attributes of the canonical graph, without copied topology."""

    catalog: dict[str, FluentDefinition] = Field(
        min_length=1,
        description="Reuse the reference vocabulary verbatim. Add a definition only for a meaning not represented there; never redefine a known value domain.",
    )
    entities: dict[str, GraphEntity] = Field(
        description="One referent table shared by all node and interface conditions, including physical objects and any explicitly declared task phases. Repeated referents keep the same identity."
    )
    nodes: dict[str, tuple[StateCondition, ...]] = Field(
        min_length=1,
        description="Conditions keyed by the canonical node IDs. Preserve current physical facts, acquired history and explicitly scoped task requirements separately in their fluent meanings.",
    )
    interfaces: dict[str, InterfaceConditions] = Field(
        description="Conditions keyed by canonical edge ID, without duplicating its endpoints, witnesses or programs."
    )
    uncertainties: tuple[str, ...] = Field(
        default=(),
        description="Unresolved meaning, omitted coordinates or identity ambiguity; not silently converted into negative facts.",
    )

    def context(self, *scopes):
        """Conjoin scope facts with their explicitly fixed entity qualifications."""
        conditions = [condition for scope in scopes for condition in scope]
        pending = {arg for condition in conditions for arg in condition.arguments}
        visited = set()
        while pending:
            entity = pending.pop()
            if entity in visited:
                continue
            visited.add(entity)
            qualifications = self.entities[entity].conditions
            conditions.extend(qualifications)
            pending.update(
                arg for condition in qualifications for arg in condition.arguments
            )
        return normalize_conditions(conditions, self.catalog, set(self.entities))

    def validate_contexts(self, endpoints):
        """Check the same conjunction at publication and matching boundaries."""
        for conditions in self.nodes.values():
            self.context(conditions)
        for key, (source, target) in endpoints.items():
            interface = self.interfaces[key]
            if source is None and not interface.entry:
                raise ValueError("known background entrance requires entry conditions")
            self.context(self.nodes.get(source, ()), interface.entry)
            self.context(self.nodes[target], interface.exit)

    @model_validator(mode="after")
    def validate_conditions(self):
        if any(
            not key.strip()
            for mapping in (self.catalog, self.entities, self.nodes, self.interfaces)
            for key in mapping
        ):
            raise ValueError("condition graph IDs must be nonempty")
        entities = set(self.entities)
        normalized_entities = {
            key: entity.model_copy(
                update={
                    "conditions": normalize_conditions(
                        entity.conditions, self.catalog, entities
                    )
                }
            )
            for key, entity in self.entities.items()
        }
        normalized_nodes = {
            key: normalize_conditions(conditions, self.catalog, entities)
            for key, conditions in self.nodes.items()
        }
        if any(not conditions for conditions in normalized_nodes.values()):
            raise ValueError("every matching node needs an explicit condition")
        normalized_interfaces = {
            key: interface.model_copy(
                update={
                    scope: normalize_conditions(
                        getattr(interface, scope), self.catalog, entities
                    )
                    for scope in ("entry", "exit")
                }
            )
            for key, interface in self.interfaces.items()
        }
        object.__setattr__(self, "entities", normalized_entities)
        object.__setattr__(self, "nodes", normalized_nodes)
        object.__setattr__(self, "interfaces", normalized_interfaces)
        return self


def merge_catalogs(*catalogs):
    """Value coordinates are comparable only under the same frozen definition."""
    result = {}
    for catalog in catalogs:
        for key, definition in catalog.items():
            if key in result and result[key] != definition:
                raise ValueError(f"conflicting definition for shared fluent {key!r}")
            result[key] = definition
    return result
