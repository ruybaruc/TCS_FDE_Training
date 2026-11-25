# Shop module

def add_item(cart, name, price, quantity=1):
    """Add an item to the cart, or increase its quantity if it already exists."""
    if quantity <= 0:
        raise ValueError('quantity must be positive')
    if name in cart:
        cart[name]['quantity'] += quantity
    else:
        cart[name] = {'price': price, 'quantity': quantity}
    return cart

def remove_item(cart, name):
    """Remove an item from the cart. Returns True if it was removed."""
    return cart.pop(name, None) is not None

def cart_total(cart):
    """Return the total price of all items in the cart."""
    return sum(item['price'] * item['quantity'] for item in cart.values())

def apply_discount(total, percent):
    """Apply a percentage discount to a total."""
    if not 0 <= percent <= 100:
        raise ValueError('percent must be between 0 and 100')
    return round(total * (1 - percent / 100), 2)
