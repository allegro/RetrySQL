REASONING_PROMPT = """You are a SQL expert. When I provide you with a SQL query, your task is to describe step-by-step how a person would create such a query. Follow the standard SQL execution order. Write the steps from the perspective of someone constructing the query. If the query includes subqueries, describe them step-by-step in the same detailed manner as the main query before referencing them. Each step should represent a distinct operation. Make the operations as granular as possible.

Here is the standard SQL execution order you should follow for your explanation. If a given clause is not present in the query, skip it without mentioning its absence:
1. FROM clause (including JOINs).
2. WHERE clause.
3. GROUP BY clause.
4. HAVING clause.
5. SELECT clause.
6. ORDER BY clause.
7. LIMIT clause.

For each query I provide:
1. Explain the query step by step in plain language.
2. Ensure that each step corresponds to one small, logical operation.
3. Use clear and concise language for each operation.
4. Each step should be provided in a single line (use single newline character between steps).

Here is an example query for your reference:
```
SELECT T1.name, T1.email, SUM(T3.amount) AS total_sales FROM Customers AS T1 INNER JOIN Orders AS T2 ON T1.customer_id = T2.customer_id LEFT JOIN OrderDetails AS T3 ON T2.order_id = T3.order_id WHERE T2.order_date >= '2023-01-01' AND T2.order_date <= (SELECT order_date FROM Orders WHERE order_id = '1' ORDER BY order_date DESC LIMIT 1) GROUP BY T1.name, T1.email ORDER BY total_sales DESC
```
The expected step-by-step breakdown for the above query:
Define the main table in the FROM clause: `FROM Customers AS T1`.
Define the first JOIN operation: `INNER JOIN`.
Define the table to join: `Orders AS T2`.
Define the join condition: `ON T1.customer_id = T2.customer_id`.
Define the second JOIN operation: `LEFT JOIN`.
Define the table to join: `OrderDetails AS T3`.
Define the join condition: `ON T2.order_id = T3.order_id`.
Define the main filtering condition in the WHERE clause: `WHERE T2.order_date >= '2023-01-01'`.
Add the additional filtering condition in the WHERE clause : `AND T2.order_date <= (subquery)`.
Define the main table in the subquery's FROM clause: `FROM Orders`.
Define the main filtering condition in the subquery's WHERE clause: `WHERE order_id = '1'`.
Select the column to be included in the subquery result: `SELECT order_date`.
Order the subquery results by the specified column: `ORDER BY order_date DESC`.
Limit the subquery results: `LIMIT 1`.
Complete the filtering condition in the WHERE clause: `AND T2.order_date <= (SELECT order_date FROM Orders WHERE order_id = '1' ORDER BY order_date DESC LIMIT 1)`.
Group the results by the specified columns: `GROUP BY T1.name, T1.email`.
Select the columns to be included in the final result: `SELECT T1.name, T1.email, SUM(T3.amount) AS total_sales`.
Order the results by the specified column: `ORDER BY total_sales DESC`.

Now, I will provide you with a query, and I expect you to respond in this format:
```
{sql_query}
```
"""