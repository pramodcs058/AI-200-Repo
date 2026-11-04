

"""Flask application demonstrating vector similarity search with PostgreSQL and pgvector."""

import json
import logging
import os

from flask import Flask, render_template, request, redirect, url_for, flash
import psycopg
from azure.identity import DefaultAzureCredential

logging.getLogger("werkzeug").setLevel(logging.WARNING)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Minimum cosine similarity required for a product match
SIMILARITY_THRESHOLD = 0.70


def get_connection():
    """Create a database connection using Microsoft Entra authentication."""
    host = os.environ.get("DB_HOST")
    dbname = os.environ.get("DB_NAME", "postgres")
    user = os.environ.get("DB_USER")

    if not host or not user:
        raise ValueError("DB_HOST and DB_USER environment variables must be set")

    credential = DefaultAzureCredential()
    token = credential.get_token(
        "https://ossrdbms-aad.database.windows.net/.default"
    )

    return psycopg.connect(
        host=host,
        dbname=dbname,
        user=user,
        password=token.token,
        sslmode="require"
    )


def load_json_file(filename):
    """Load products from a JSON file."""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("products", [])


def get_products():
    """Retrieve all products from the database."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, name, category, description, price
                    FROM products
                    ORDER BY name
                """)
                return [
                    {
                        "id": r[0],
                        "name": r[1],
                        "category": r[2],
                        "description": r[3],
                        "price": r[4]
                    }
                    for r in cur.fetchall()
                ]
    except Exception:
        return []


def get_new_products():
    """Load new products from JSON that aren't in the database."""
    new_products = load_json_file("new_products.json")
    existing = {p["name"] for p in get_products()}
    return [(i, p) for i, p in enumerate(new_products) if p["name"] not in existing]


@app.route("/")
def index():
    products = get_products()
    new_products = get_new_products()
    return render_template(
        "index.html",
        products=products,
        new_products=new_products,
        results=None
    )


@app.route("/load-data", methods=["POST"])
def load_data():
    """Load sample products into the database."""
    try:
        products = load_json_file("sample_products.json")

        with get_connection() as conn:
            with conn.cursor() as cur:
                for product in products:
                    cur.execute(
                        "SELECT id FROM products WHERE name = %s",
                        (product["name"],)
                    )
                    if cur.fetchone():
                        continue

                    embedding_str = "[" + ",".join(
                        str(x) for x in product["embedding"]
                    ) + "]"

                    cur.execute("""
                        INSERT INTO products
                            (name, category, description, price, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        product["name"],
                        product["category"],
                        product["description"],
                        product["price"],
                        embedding_str
                    ))

                conn.commit()

        flash(f"Successfully loaded {len(products)} sample products!", "success")

    except Exception as e:
        flash(f"Error loading data: {str(e)}", "error")

    return redirect(url_for("index"))


@app.route("/search", methods=["POST"])
def search():
    """Find products whose cosine similarity is at least 0.70."""
    product_id = request.form.get("product_id")

    if not product_id:
        flash("Please select a product", "error")
        return redirect(url_for("index"))

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, name, category, description, price, embedding
                    FROM products
                    WHERE id = %s
                """, (product_id,))

                row = cur.fetchone()

                if not row:
                    flash("Product not found", "error")
                    return redirect(url_for("index"))

                searched_product = {
                    "id": row[0],
                    "name": row[1],
                    "category": row[2],
                    "description": row[3],
                    "price": row[4]
                }

                embedding = row[5]

                # pgvector <=> returns cosine distance.
                # cosine similarity = 1 - cosine distance.
                cur.execute("""
                    SELECT
                        id,
                        name,
                        category,
                        description,
                        price,
                        1 - (embedding <=> %s) AS similarity
                    FROM products
                    WHERE id != %s
                      AND 1 - (embedding <=> %s) >= %s
                    ORDER BY similarity DESC
                    LIMIT 5
                """, (
                    embedding,
                    product_id,
                    embedding,
                    SIMILARITY_THRESHOLD
                ))

                results = [
                    {
                        "id": r[0],
                        "name": r[1],
                        "category": r[2],
                        "description": r[3],
                        "price": r[4],
                        "similarity": float(r[5]),
                        # Keep distance for compatibility with the existing index.html template.
                        # distance = 1 - cosine similarity
                        "distance": 1.0 - float(r[5])
                    }
                    for r in cur.fetchall()
                ]

        products = get_products()
        new_products = get_new_products()

        return render_template(
            "index.html",
            products=products,
            new_products=new_products,
            results=results,
            searched_product=searched_product
        )

    except Exception as e:
        flash(f"Error searching: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/add-product", methods=["POST"])
def add_product():
    """Add a new product from new_products.json."""
    product_index = request.form.get("product_index")

    if product_index is None or product_index == "":
        flash("Please select a product to add", "error")
        return redirect(url_for("index"))

    try:
        new_products = load_json_file("new_products.json")
        product = new_products[int(product_index)]

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM products WHERE name = %s",
                    (product["name"],)
                )

                if cur.fetchone():
                    flash(
                        f"Product '{product['name']}' already exists",
                        "error"
                    )
                    return redirect(url_for("index"))

                embedding_str = "[" + ",".join(
                    str(x) for x in product["embedding"]
                ) + "]"

                cur.execute("""
                    INSERT INTO products
                        (name, category, description, price, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    product["name"],
                    product["category"],
                    product["description"],
                    product["price"],
                    embedding_str
                ))

                conn.commit()

        flash(f"Successfully added '{product['name']}'!", "success")

    except Exception as e:
        flash(f"Error adding product: {str(e)}", "error")

    return redirect(url_for("index"))


if __name__ == "__main__":
    print("Dashboard running at http://127.0.0.1:5000", flush=True)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
