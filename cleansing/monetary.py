"""cleansing current-price inference and pure monetary helpers.

This module never modifies source records or writes a CSV. Inference requires
rows whose carts AND observed order prices are independently known to be valid.
For Retail, use the selected outlier-batch rows: only delivery charges in that
batch can be anomalous (data contract section 2, batch guarantees).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CartItem:
    product: str
    quantity: int


def parse_cart(cart_json: str) -> tuple[CartItem, ...]:
    """Read a JSON cart without eval or changes to the original string."""
    if not isinstance(cart_json, str):
        raise ValueError("shopping_cart must be a JSON string")
    try:
        raw = json.loads(cart_json)
    except json.JSONDecodeError as exc:
        raise ValueError("shopping_cart is not valid JSON") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError("Expected a non-empty JSON array of cart items")
    items = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each cart item must be an object")
        product, quantity = item.get("product"), item.get("quantity")
        if not isinstance(product, str) or not product.strip():
            raise ValueError("Each product must be a non-empty string")
        # bool is a subclass of int in Python; it is not a valid quantity.
        if type(quantity) is not int or quantity <= 0:
            raise ValueError("Each quantity must be a positive integer")
        items.append(CartItem(product, quantity))
    return tuple(items)


def _number(value: object, label: str) -> float:
    """Accept CSV numeric strings, but reject booleans/non-finite values."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def cart_price(cart_json: str, unit_prices: Mapping[str, float]) -> float:
    """Round each quantity * unit price, sum, then round the sum (spec 3.2).

    Keep individual cart entries separate. Do not round the unit price first,
    and do not aggregate repeated product entries before rounding line amounts.
    Returns a calculated value only; the caller decides whether a source field
    needs repair based on independent diagnosis.
    """
    line_amounts = []
    for item in parse_cart(cart_json):
        if item.product not in unit_prices:
            raise ValueError(f"No inferred price for product: {item.product!r}")
        price = _number(unit_prices[item.product], "unit price")
        if price <= 0:
            raise ValueError("Unit prices must be positive")
        amount = _number(item.quantity * price, "line amount")
        line_amounts.append(round(amount, 2))
    return round(_number(sum(line_amounts), "cart sum"), 2)


def order_total(order_price: object, coupon_discount: object,
                delivery_charges: object) -> float:
    """Apply the published formula in its published operation order.

    Uses Python floats and round, not Decimal arithmetic. Do not round the
    discounted subtotal. The coupon applies only to items; no GST is added.
    This function does not overwrite an observed total.
    """
    price = _number(order_price, "order_price")
    discount = _number(coupon_discount, "coupon_discount")
    delivery = _number(delivery_charges, "delivery_charges")
    if price < 0 or delivery < 0 or not 0 <= discount <= 100:
        raise ValueError("Require nonnegative money and a discount in [0, 100]")
    total = price * (1 - discount / 100) + delivery
    return round(_number(total, "order_total"), 2)


@dataclass(frozen=True)
class PriceInference:
    products: tuple[str, ...]
    prices: dict[str, float]
    raw_coefficients: tuple[float, ...]
    training_count: int
    rank: int
    condition_number: float
    max_distance_from_cent: float
    max_linear_residual: float
    max_cart_residual: float


def infer_current_prices(trusted_rows: Sequence[Mapping[str, str]],
                         numerical_tolerance: float = 1e-7) -> PriceInference:
    """Fit X p = y using only the supplied, trusted training observations.

    X[i,j] is product j's quantity in order i; y[i] is its observed order_price.
    No intercept: an order price is the sum of its purchased item amounts.
    Product names are discovered from training carts only. Identifiers are not
    features, and no dirty prices or delivery charges enter these equations.

    For this package the solution lies within floating-point noise of cents.
    Snap only if every coefficient is within numerical_tolerance of a cent,
    then recheck the required rounded cart calculation on all training rows.
    This is an observed Retail property, not a claim that every possible cleansing
    package must have cent-valued unit prices. Reject unsupported snapping.
    numerical_tolerance removes solver noise; it is NOT a export tolerance.
    """
    if not trusted_rows:
        raise ValueError("No trusted training observations supplied")
    if not math.isfinite(numerical_tolerance) or numerical_tolerance <= 0:
        raise ValueError("numerical_tolerance must be positive and finite")
    carts = [parse_cart(row["shopping_cart"]) for row in trusted_rows]
    products = tuple(sorted({item.product for cart in carts for item in cart}))
    column = {product: j for j, product in enumerate(products)}
    X = np.zeros((len(carts), len(products)), dtype=float)
    for i, cart in enumerate(carts):
        for item in cart:
            X[i, column[item.product]] += item.quantity
    y = np.array([_number(row["order_price"], "observed order_price")
                  for row in trusted_rows], dtype=float)
    if np.any(y <= 0):
        raise ValueError("Training order prices must be positive")
    raw, _, rank, singular_values = np.linalg.lstsq(X, y, rcond=None)
    if rank != len(products):
        raise ValueError("Prices are not identifiable: training matrix is rank deficient")
    snapped = np.array([round(float(value), 2) for value in raw])
    gap = float(np.max(np.abs(raw - snapped)))
    if gap > numerical_tolerance:
        raise ValueError("Evidence does not support cent-valued price normalisation")
    if np.any(snapped <= 0):
        raise ValueError("Inferred unit prices must be positive")
    prices = dict(zip(products, map(float, snapped)))
    linear_error = float(np.max(np.abs(X @ snapped - y)))
    cart_error = max(abs(cart_price(row["shopping_cart"], prices) - observed)
                     for row, observed in zip(trusted_rows, y))
    if max(linear_error, cart_error) > numerical_tolerance:
        raise ValueError("Trusted training equations are inconsistent with inferred prices")
    return PriceInference(
        products=products, prices=prices, raw_coefficients=tuple(map(float, raw)),
        training_count=len(trusted_rows), rank=int(rank),
        condition_number=float(singular_values[0] / singular_values[-1]),
        max_distance_from_cent=gap, max_linear_residual=linear_error,
        max_cart_residual=float(cart_error),
    )
