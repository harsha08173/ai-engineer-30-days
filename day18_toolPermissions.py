def search_attractions(destination: str):
    # Read-only action: safe to execute
    return f"Found attractions in {destination}"


def book_hotel(hotel: str, approved: bool = False):
    # Enforce approval in application code, not just in the prompt
    if not approved:
        return "Booking blocked: user approval required."

    # In a real app, the booking API call goes here
    return f"Booking request submitted for {hotel}"


print(search_attractions("Goa"))

# Attempt booking without approval
print(book_hotel("Goa Beach Resort"))

# Booking after explicit approval
print(book_hotel("Goa Beach Resort", approved=True))