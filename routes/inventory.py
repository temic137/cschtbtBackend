from flask import Blueprint, jsonify
from extensions import inventory_collection
from bson.json_util import dumps

inventory_bp = Blueprint('inventory', __name__)

@inventory_bp.route('/api/inventory', methods=['GET'])
def get_inventory():
    try:
        inventory = list(inventory_collection.find({}, {'_id': 0}))
        return dumps(inventory), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        return jsonify({"error": f"Failed to fetch inventory data: {str(e)}"}), 500