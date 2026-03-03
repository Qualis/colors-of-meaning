---
name: Self-Documenting Code Refactorer
description: This skill should be used when the user asks to "remove comments", "make code self-documenting", "refactor to eliminate comments", "improve code clarity", "make this clearer without comments", or mentions the no-comments rule. This skill helps refactor code to be self-explanatory through expressive naming instead of comments.
version: 1.0.0
---

# Self-Documenting Code Refactorer

## Overview

This project has a **strict NO COMMENTS policy**. All code must be self-documenting through expressive naming and clear structure. This skill guides you through identifying comments and refactoring code to eliminate the need for them.

## Core Principle

> "Code should be so clear that comments are unnecessary. If you need a comment, refactor the code to be clearer instead."

## When to Use This Skill

Use this skill when:
- You find comments in code that explain what or why
- Code needs clarification but you're tempted to add a comment
- Reviewing code for the no-comments policy
- Refactoring unclear code to be self-explanatory

## The NO COMMENTS Rule

### Absolutely Forbidden
- Implementation comments explaining what code does
- TODO comments
- Inline explanations
- Block comments describing logic
- Commented-out code

### The Only Exception
- Docstrings are NOT considered comments (but this project doesn't use them either)
- Project prefers self-documenting code over docstrings

## Refactoring Strategies

### Strategy 1: Extract to Well-Named Function

**Before (with comment):**
```python
def process_order(order):
    # Calculate total with tax
    subtotal = sum(item.price for item in order.items)
    tax = subtotal * 0.08
    total = subtotal + tax
    return total
```

**After (self-documenting):**
```python
def process_order(order):
    return calculate_order_total_with_tax(order)

def calculate_order_total_with_tax(order):
    subtotal = calculate_subtotal(order)
    tax = calculate_tax(subtotal)
    return subtotal + tax

def calculate_subtotal(order):
    return sum(item.price for item in order.items)

def calculate_tax(subtotal):
    TAX_RATE = 0.08
    return subtotal * TAX_RATE
```

**Key Insight:** The function name explains what the code does better than any comment.

### Strategy 2: Expressive Variable Names

**Before (with comment):**
```python
def get_users(db):
    # Only active users in last 30 days
    cutoff = datetime.now() - timedelta(days=30)
    users = db.query(User).filter(User.last_login > cutoff, User.active == True)
    return users
```

**After (self-documenting):**
```python
def get_recently_active_users(db):
    thirty_days_ago = datetime.now() - timedelta(days=30)
    users_active_in_last_thirty_days = db.query(User).filter(
        User.last_login > thirty_days_ago,
        User.active == True
    )
    return users_active_in_last_thirty_days
```

**Key Insight:** Variable names carry the information that was in the comment.

### Strategy 3: Boolean Variable for Complex Conditions

**Before (with comment):**
```python
def can_process(order):
    # Check if order is valid, paid, and in stock
    if order.items and order.payment_status == "paid" and all(item.in_stock for item in order.items):
        return True
    return False
```

**After (self-documenting):**
```python
def can_process(order):
    order_has_items = len(order.items) > 0
    payment_is_complete = order.payment_status == "paid"
    all_items_in_stock = all(item.in_stock for item in order.items)

    order_is_ready_for_processing = (
        order_has_items and
        payment_is_complete and
        all_items_in_stock
    )

    return order_is_ready_for_processing
```

**Key Insight:** Break complex conditions into named boolean variables.

### Strategy 4: Extract Magic Numbers to Named Constants

**Before (with comment):**
```python
def calculate_discount(amount):
    # 15% discount for premium customers
    return amount * 0.85
```

**After (self-documenting):**
```python
def calculate_premium_customer_discount(amount):
    PREMIUM_DISCOUNT_RATE = 0.15
    discount_multiplier = 1 - PREMIUM_DISCOUNT_RATE
    return amount * discount_multiplier
```

**Key Insight:** Named constants and descriptive function names replace comments.

### Strategy 5: Small, Single-Purpose Functions

**Before (with comment):**
```python
def handle_user_request(user, request):
    # Validate user permissions
    if not user.has_permission("admin"):
        raise PermissionError()

    # Log the request
    logger.info(f"User {user.id} made request")

    # Process the request
    result = process(request)

    # Update metrics
    metrics.increment("requests_processed")

    return result
```

**After (self-documenting):**
```python
def handle_user_request(user, request):
    ensure_user_has_admin_permission(user)
    log_user_request(user)
    result = process(request)
    increment_request_metrics()
    return result

def ensure_user_has_admin_permission(user):
    if not user.has_permission("admin"):
        raise PermissionError("Admin permission required")

def log_user_request(user):
    logger.info(f"User {user.id} made request")

def increment_request_metrics():
    metrics.increment("requests_processed")
```

**Key Insight:** Each section becomes a well-named function that documents itself.

## Refactoring Process

### Step 1: Identify Comments

Scan code for:
- Lines starting with `#`
- Multi-line comments with `"""` or `'''`
- TODO, FIXME, HACK, NOTE markers
- Commented-out code blocks

### Step 2: Understand the Comment's Purpose

Ask yourself:
- **What does this comment explain?** (The behavior/logic)
- **Why does this code need explanation?** (Too complex? Unclear naming?)
- **What information does the comment provide?** (Business rule? Edge case? Calculation?)

### Step 3: Choose Refactoring Strategy

Based on the comment type:

| Comment Type | Refactoring Strategy |
|--------------|---------------------|
| Explains what code does | Extract to well-named function |
| Describes complex logic | Break into smaller functions with descriptive names |
| Clarifies variable purpose | Rename variable to be more descriptive |
| Explains magic number | Extract to named constant |
| Describes condition | Extract to boolean variable with descriptive name |
| Documents algorithm | Use algorithm name in function name |
| Explains business rule | Encode rule in function/variable names |

### Step 4: Refactor

Apply the chosen strategy:
1. Extract code to new function/variable/constant
2. Give it a descriptive, self-explanatory name
3. Remove the comment
4. Verify tests still pass

### Step 5: Verify Self-Documentation

Ask yourself:
- Can I understand what this code does without comments?
- Are function names action-oriented and specific?
- Are variable names descriptive enough?
- Are complex conditions broken down?
- Would a new developer understand this code?

If yes to all, the refactoring is complete.

## Naming Conventions

### Functions
- Use verb phrases: `calculate_total`, `validate_user`, `send_email`
- Be specific: `get_active_users_from_last_month` > `get_users`
- Include business context: `apply_premium_discount` > `apply_discount`

### Variables
- Use descriptive nouns: `active_user_count` > `count`
- Include units: `timeout_in_seconds` > `timeout`
- Express purpose: `users_eligible_for_discount` > `users`

### Constants
- Use UPPER_CASE: `MAX_RETRY_ATTEMPTS = 3`
- Be explicit: `DEFAULT_PAGE_SIZE = 20` > `DEFAULT = 20`
- Include context: `PREMIUM_DISCOUNT_PERCENTAGE = 15`

### Boolean Variables
- Use predicates: `is_valid`, `has_permission`, `can_process`
- Be affirmative: `is_active` > `is_not_inactive`
- Express state: `user_is_authenticated` > `authenticated`

## Common Refactoring Patterns

### Pattern 1: Comment Explaining "Why"

**Before:**
```python
# We need to wait 5 seconds because the external API has rate limiting
time.sleep(5)
```

**After:**
```python
EXTERNAL_API_RATE_LIMIT_DELAY_IN_SECONDS = 5

def wait_for_api_rate_limit():
    time.sleep(EXTERNAL_API_RATE_LIMIT_DELAY_IN_SECONDS)

wait_for_api_rate_limit()
```

### Pattern 2: Comment Listing Steps

**Before:**
```python
def checkout(cart):
    # 1. Validate cart
    # 2. Calculate total
    # 3. Process payment
    # 4. Send confirmation
    pass
```

**After:**
```python
def checkout(cart):
    validate_cart(cart)
    total = calculate_cart_total(cart)
    process_payment(total)
    send_order_confirmation()
```

### Pattern 3: Comment for Complex Condition

**Before:**
```python
# User can edit if they're the owner or an admin and the document isn't locked
if (user.id == doc.owner_id or user.role == "admin") and not doc.is_locked:
    allow_edit()
```

**After:**
```python
user_is_owner = user.id == doc.owner_id
user_is_admin = user.role == "admin"
document_is_unlocked = not doc.is_locked

user_can_edit_document = (
    (user_is_owner or user_is_admin) and
    document_is_unlocked
)

if user_can_edit_document:
    allow_edit()
```

### Pattern 4: Comment for Algorithm

**Before:**
```python
# Binary search implementation
def search(arr, target):
    left = 0
    right = len(arr) - 1
    # ... implementation
```

**After:**
```python
def binary_search(sorted_array, target_value):
    left_boundary = 0
    right_boundary = len(sorted_array) - 1
    # Implementation clearly shows binary search pattern
```

## Warning Signs Code Needs Refactoring

If you find yourself wanting to add comments for these reasons, refactor instead:

1. **"This is a workaround for..."** → Create `workaround_for_X` function
2. **"This calculates..."** → Function name should say what it calculates
3. **"Loop through and..."** → Extract loop to `process_each_X` function
4. **"Check if..."** → Extract to `is_X` or `has_Y` function
5. **"Initialize..."** → Use `create_X` or `build_Y` function
6. **"Clean up..."** → Extract to `cleanup_X` function
7. **"Handle edge case where..."** → Extract to `handle_edge_case_X` function

## Code Review Checklist

When reviewing code for the no-comments policy:

- [ ] Zero comments exist in code
- [ ] Function names clearly describe what they do
- [ ] Variable names clearly describe what they contain
- [ ] Complex conditions are broken into named booleans
- [ ] Magic numbers are extracted to named constants
- [ ] Each function does one thing
- [ ] Business logic is encoded in names, not comments
- [ ] Code reads like well-written prose

## Examples from This Project

### Good: Domain Model (No Comments Needed)

```python
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID

class Coconut(BaseModel):
    id: Optional[UUID] = Field(default=None, alias="id")
```

The code is self-explanatory. No comments needed.

### Good: Use Case (Clear Intent)

```python
class GetCoconutUseCase:
    def __init__(self, query_repository: CoconutQueryRepository):
        self._query_repository = query_repository

    def execute(self, id: UUID) -> Coconut:
        return self._query_repository.read(id)
```

Function and class names make the purpose obvious.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Obvious Comments

```python
# Increment counter
counter += 1

# Return the result
return result
```

These comments state what the code obviously does. Just remove them.

### Anti-Pattern 2: Commented-Out Code

```python
def process():
    do_something()
    # old_method()  # ❌ Never leave commented code
    # alternative_approach()  # ❌ Use git for history
```

Use version control (git) for code history. Delete commented code.

### Anti-Pattern 3: TODO Comments

```python
# TODO: Add error handling  # ❌ Either do it now or create a ticket
def risky_operation():
    pass
```

Either implement it immediately or create a proper issue/ticket. Don't leave TODOs.

## Dealing with Truly Complex Logic

If logic is genuinely complex (algorithms, mathematical formulas):

1. **Use descriptive function name**: `calculate_compound_interest_with_monthly_contributions`
2. **Break into smaller functions**: Each step becomes a function
3. **Use domain language**: Name functions after business concepts
4. **Extract constants**: Make formulas clear through named values

**Example:**
```python
def calculate_compound_interest_with_monthly_contributions(
    principal_amount,
    annual_interest_rate,
    years,
    monthly_contribution
):
    number_of_compounding_periods_per_year = 12
    total_number_of_periods = years * number_of_compounding_periods_per_year

    future_value_of_principal = calculate_compound_interest(
        principal_amount,
        annual_interest_rate,
        number_of_compounding_periods_per_year,
        total_number_of_periods
    )

    future_value_of_contributions = calculate_future_value_of_series(
        monthly_contribution,
        annual_interest_rate,
        number_of_compounding_periods_per_year,
        total_number_of_periods
    )

    return future_value_of_principal + future_value_of_contributions
```

The formulas are still complex, but each piece is named and understandable.

## Tools for Detection

Use these tools to find comments:

```bash
# Find Python comments in source
grep -r "^\s*#" src/

# Find TODO/FIXME markers
grep -r "TODO\|FIXME\|HACK" src/

# Count comments
grep -r "^\s*#" src/ | wc -l
```

Expected output: **0 comments**

## Remember

> "Any fool can write code that a computer can understand. Good programmers write code that humans can understand."
> — Martin Fowler

In this project, we take it further: write code so clear that comments become redundant.

## References

See the `references/` directory for:
- Before/after refactoring examples
- Common naming patterns
- Project code examples without comments
