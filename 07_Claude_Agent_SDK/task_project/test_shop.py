# Tests for shop module

from shop import add_item, remove_item, cart_total, apply_discount

def test_add_item():
    """Test adding a new item to the cart."""
    cart = add_item({}, 'apple', 1.5, 2)
    assert cart == {'apple': {'price': 1.5, 'quantity': 2}}

def test_add_existing_item_increases_quantity():
    """Test that adding an existing item increases its quantity."""
    cart = add_item({}, 'apple', 1.5, 2)
    cart = add_item(cart, 'apple', 1.5, 3)
    assert cart['apple']['quantity'] == 5

def test_remove_item():
    """Test removing an item from the cart."""
    cart = add_item({}, 'apple', 1.5)
    assert remove_item(cart, 'apple') == True
    assert 'apple' not in cart

def test_remove_missing_item():
    """Test removing an item that is not in the cart."""
    assert remove_item({}, 'banana') == False

def test_cart_total():
    """Test the total price of the cart."""
    cart = add_item({}, 'apple', 1.5, 2)
    cart = add_item(cart, 'bread', 3.0, 1)
    assert cart_total(cart) == 6.0

def test_apply_discount():
    """Test applying a percentage discount."""
    assert apply_discount(100, 10) == 90.0
