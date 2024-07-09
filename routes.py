from flask import render_template, request, redirect, url_for
from utils import add_new_employee, view_existing_employees, authenticate_employee_real_time, delete_embeddings_for_employee

def setup_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/add', methods=['GET', 'POST'])
    def add_employee():
        if request.method == 'POST':
            add_new_employee()
            return redirect(url_for('view_employees'))
        return render_template('add_employee.html')

    @app.route('/view')
    def view_employees():
        employees = view_existing_employees()
        return render_template('view_employees.html', employees=employees)

    @app.route('/authenticate')
    def authenticate_employee():
        authenticate_employee_real_time()
        return redirect(url_for('index'))  # Assuming redirecting to home after authentication

    @app.route('/delete', methods=['GET', 'POST'])
    def delete_employee():
        if request.method == 'POST':
            delete_embeddings_for_employee()
            return redirect(url_for('index'))  # Redirect to home after deletion
        return render_template('delete_employee.html')  # Assuming you have a template for deletion

