from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from supabase import create_client, Client

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Change this in production

# Supabase config
SUPABASE_URL = "https://hatdisgzirzmunerdnnu.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImhhdGRpc2d6aXJ6bXVuZXJkbm51Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTE2NzczNCwiZXhwIjoyMDcwNzQzNzM0fQ.ZBqV9FNWBh_CyUpJid9JFEV2nmstmD06NKM4Lx6JZsA"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        # Query Supabase for user
        response = supabase.table("users").select("*").eq("email", email).execute()

        if response.data:
            user = response.data[0]
            if user["password"] == password:  # NOTE: use hashing in production!
                session["user"] = {
                    "email": user["email"],
                    "name": user["name"],
                    "role": user["role"]
                }
                flash("Login successful!", "success")
                return redirect(url_for("dashboard"))
            else:
                flash("Invalid password!", "danger")
        else:
            flash("User not found!", "danger")

    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", user=session["user"])

@app.route("/stock-book")
def stock_book():
    response = supabase.table("stock_book").select("*").execute()
    data = response.data if response.data else []
    return render_template("stock_book.html", data=data)

@app.route("/purchase-book")
def purchase_book():
    response = supabase.table("purchase_book").select("*").execute()
    data = response.data if response.data else []
    return render_template("purchase_book.html", data=data)

@app.route("/sale-book")
def sale_book():
    response = supabase.table("sales_book").select("*").execute()
    data = response.data if response.data else []
    return render_template("sale_book.html", data=data)

@app.route("/add-sale", methods=["GET", "POST"])
def add_sale():
    # Get last invoice number
    last_invoice_resp = supabase.table("sales_book").select("invoice_no").order("id", desc=True).limit(1).execute()
    next_invoice = 1
    if last_invoice_resp.data:
        try:
            last_invoice_no = last_invoice_resp.data[0].get("invoice_no", "0")
            next_invoice = int(last_invoice_no) + 1
        except ValueError:
            next_invoice = 1

    prefilled_data = {}  # To hold prefilled price and quantity_type

    if request.method == "POST":
        date = request.form.get("date")
        invoice_no = request.form.get("invoice_no")
        hsn = request.form.get("hsn")
        item = request.form.get("item")
        quantity = request.form.get("quantity")
        price = request.form.get("price")
        amount = request.form.get("amount")

        # Validate required fields
        if not all([date, invoice_no, hsn, item, quantity, price, amount]):
            flash("All fields are required.", "danger")
            return render_template("add_sale.html", next_invoice=next_invoice, prefilled_data=prefilled_data)

        try:
            quantity = float(quantity)
            price = float(price)
            amount = float(amount)
        except ValueError:
            flash("Quantity, price, and amount must be numeric.", "danger")
            return render_template("add_sale.html", next_invoice=next_invoice, prefilled_data=prefilled_data)

        # Insert into sales_book
        response = supabase.table("sales_book").insert({
            "date": date,
            "invoice_no": invoice_no,
            "item": item,
            "quantity": quantity,
            "price": price,
            "amount": amount
        }).execute()

        if response.data:
            flash("Sale added successfully!", "success")
            return redirect(url_for("sale_book"))
        else:
            flash("Failed to add sale.", "danger")

    elif request.method == "GET":
        # Optional: prefill price if HSN is provided as query param
        hsn = request.args.get("hsn")
        if hsn:
            product_resp = supabase.table("stock_items").select("*").eq("hsn", hsn).execute()
            if product_resp.data:
                product = product_resp.data[0]
                prefilled_data = {
                    "price": product.get("price", 0),
                    "quantity_type": product.get("quantity_type", "")
                }

    return render_template("add_sale.html", next_invoice=next_invoice, prefilled_data=prefilled_data)

# API route to fetch product details by HSN
@app.route("/get-product/<hsn>")
def get_product(hsn):
    # Fetch product from stock_items table where HSN matches
    product_resp = supabase.table("stock_items").select("*").eq("hsn", hsn).execute()
    
    if product_resp.data and len(product_resp.data) > 0:
        product = product_resp.data[0]
        return jsonify({
            "item": product.get("item", ""),
            "hsn": product.get("hsn", ""),
            "quantity_type": product.get("quantity_type", ""),
            "price": product.get("price", 0)
        })
    
    # If no product found, return 404 error with message
    return jsonify({"error": "Product not found"}), 404

# Route for Add Purchase Page
@app.route("/add-purchase", methods=["GET", "POST"])
def add_purchase():
    # Get last invoice number
    last_invoice_resp = supabase.table("purchase_book").select("invoice_no").order("id", desc=True).limit(1).execute()
    next_invoice = 1
    if last_invoice_resp.data:
        try:
            last_invoice_no = last_invoice_resp.data[0].get("invoice_no", "0")
            next_invoice = int(last_invoice_no) + 1
        except ValueError:
            next_invoice = 1

    if request.method == "POST":
        date = request.form.get("date")
        invoice_no = request.form.get("invoice_no")
        hsn = request.form.get("hsn")
        name = request.form.get("name")
        qty = request.form.get("qty")
        basic = request.form.get("basic")
        sgst = request.form.get("sgst")
        cgst = request.form.get("cgst")
        igst = request.form.get("igst")
        round_off = request.form.get("round_off")
        total = request.form.get("total")

        # Validate required fields
        if not all([date, invoice_no, hsn, name, qty, basic, total]):
            flash("All required fields must be filled.", "danger")
            return render_template("add_purchase.html", next_invoice=next_invoice)

        try:
            qty = float(qty)
            basic = float(basic)
            sgst = float(sgst or 0)
            cgst = float(cgst or 0)
            igst = float(igst or 0)
            round_off = float(round_off or 0)
            total = float(total)
        except ValueError:
            flash("Quantity, amounts, and taxes must be numeric.", "danger")
            return render_template("add_purchase.html", next_invoice=next_invoice)

        # Insert into purchase_book
        response = supabase.table("purchase_book").insert({
            "date": date,
            "invoice_no": invoice_no,
            "hsn": hsn,
            "name": name,
            "qty": qty,
            "basic": basic,
            "sgst": sgst,
            "cgst": cgst,
            "igst": igst,
            "round_off": round_off,
            "total": total
        }).execute()

        if response.data:
            flash("Purchase added successfully!", "success")
            return redirect(url_for("purchase_book"))  # Replace with your listing route
        else:
            flash("Failed to add purchase.", "danger")

    return render_template("add_purchase.html", next_invoice=next_invoice)


# API to fetch shop names for suggestions
@app.route("/get-shop-names")
def get_shop_names():
    resp = supabase.table("purchase_book").select("name").execute()
    names = list({row["name"] for row in resp.data if row.get("name")})
    return jsonify(names)


# API to fetch product details by HSN
@app.route("/get-purchase-product/<hsn>")
def get_purchase_product(hsn):
    product_resp = supabase.table("stock_items").select("*").eq("hsn", hsn).execute()
    if product_resp.data:
        product = product_resp.data[0]
        return jsonify({
            "item": product.get("item", ""),
            "hsn": product.get("hsn", ""),
            "price": product.get("price", 0)
        })
    return jsonify({"error": "Product not found"}), 404

# Optional: API route to fetch all HSNs for autocomplete
@app.route("/get-all-hsn")
def get_all_hsn():
    response = supabase.table("stock_items").select("hsn").execute()
    if response.data:
        return jsonify(response.data)  # [{"hsn": "2523"}, {"hsn": "2524"}, ...]
    return jsonify([])  # empty list if no data

@app.route("/add-item")
def add_item():
    return render_template("add_item.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully!", "info")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
