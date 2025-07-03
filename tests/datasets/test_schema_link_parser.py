from upath import UPath

from text2sql.datasets.domain.model import (
    DatabaseSchema,
    SchemaColumn,
    SchemaLink,
    SchemaLinkingDataSample,
    Text2SqlDataSample,
)
from text2sql.datasets.schema_linking.parsers.schema_link_parser import SchemaLinkParser


class TestSchemaLinkParser:
    def test_queries_without_joins(self):
        # GIVEN
        ground_truth_data = [
            Text2SqlDataSample(
                question_id=1,
                question="What is the average age of people?",
                database_name="db_1",
                database_path=UPath("db_1/db_1.sqlite"),
                sql_query="select avg(age) from person",
            ),
            Text2SqlDataSample(
                question_id=2,
                question="Which species are black in color?",
                database_name="db_1",
                database_path=UPath("db_1/db_1.sqlite"),
                sql_query='select species from animal where color = "black"',
            ),
        ]

        database_metadata = [
            DatabaseSchema(
                database_name="db_1",
                tables={
                    "person": [SchemaColumn(column_name="name"), SchemaColumn(column_name="age")],
                    "animal": [SchemaColumn(column_name="species"), SchemaColumn(column_name="color")],
                },
            )
        ]

        # WHEN
        schema_links = SchemaLinkParser().run(
            bird_ground_truth_data=ground_truth_data,
            bird_metadata=database_metadata,
        )

        # THEN
        assert schema_links == [
            SchemaLinkingDataSample(
                question_id=1,
                question="What is the average age of people?",
                schema_links=[SchemaLink(table_name="person", columns=["age"])],
            ),
            SchemaLinkingDataSample(
                question_id=2,
                question="Which species are black in color?",
                schema_links=[SchemaLink(table_name="animal", columns=["color", "species"])],
            ),
        ]

    def test_queries_with_joins_no_aliases(self):
        ground_truth_data = [
            Text2SqlDataSample(
                question_id=1,
                question="How many people older than 50 own a dog?",
                database_name="db_1",
                database_path=UPath("db_1/db_1.sqlite"),
                sql_query="select count(name) from animal inner join person on animal.owner = person.name "
                'where age > 50 and species = "dog"',
            )
        ]

        database_metadata = [
            DatabaseSchema(
                database_name="db_1",
                tables={
                    "person": [SchemaColumn(column_name="name"), SchemaColumn(column_name="age")],
                    "animal": [
                        SchemaColumn(column_name="species"),
                        SchemaColumn(column_name="color"),
                        SchemaColumn(column_name="owner"),
                    ],
                },
            )
        ]

        # WHEN
        schema_links = SchemaLinkParser().run(
            bird_ground_truth_data=ground_truth_data,
            bird_metadata=database_metadata,
        )

        # THEN
        assert schema_links == [
            SchemaLinkingDataSample(
                question_id=1,
                question="How many people older than 50 own a dog?",
                schema_links=[
                    SchemaLink(table_name="person", columns=["age", "name"]),
                    SchemaLink(table_name="animal", columns=["owner", "species"]),
                ],
            )
        ]

    def test_queries_with_joins_and_aliases(self):
        ground_truth_data = [
            Text2SqlDataSample(
                question_id=1,
                question="How many people older than 50 own a dog?",
                database_name="db_1",
                database_path=UPath("db_1/db_1.sqlite"),
                sql_query="select count(t1.name) from animal as t2 inner join person as t1 on t2.owner = t1.name "
                'where t1.age > 50 and t2.species = "dog"',
            )
        ]

        database_metadata = [
            DatabaseSchema(
                database_name="db_1",
                tables={
                    "person": [SchemaColumn(column_name="name"), SchemaColumn(column_name="age")],
                    "animal": [
                        SchemaColumn(column_name="species"),
                        SchemaColumn(column_name="color"),
                        SchemaColumn(column_name="owner"),
                    ],
                },
            )
        ]

        # WHEN
        schema_links = SchemaLinkParser().run(
            bird_ground_truth_data=ground_truth_data,
            bird_metadata=database_metadata,
        )

        # THEN
        assert schema_links == [
            SchemaLinkingDataSample(
                question_id=1,
                question="How many people older than 50 own a dog?",
                schema_links=[
                    SchemaLink(table_name="person", columns=["age", "name"]),
                    SchemaLink(table_name="animal", columns=["owner", "species"]),
                ],
            )
        ]
