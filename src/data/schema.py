
SOURCE_COLUMNS = [
    "IsCanceled",
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateMonth",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "ReservedRoomType",
    "AssignedRoomType",
    "BookingChanges",
    "DepositType",
    "Agent",
    "Company",
    "DaysInWaitingList",
    "CustomerType",
    "ADR",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
    "ReservationStatus",
    "ReservationStatusDate",
]

NUMERIC_COLUMNS = [
    "IsCanceled",
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
    "BookingChanges",
    "DaysInWaitingList",
    "ADR",
    "RequiredCarParkingSpaces",
    "TotalOfSpecialRequests",
]


# All categorical columns in the original H2 dataset.
CATEGORICAL_COLUMNS = [
    "ArrivalDateMonth",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "ReservedRoomType",
    "AssignedRoomType",
    "DepositType",
    "Agent",
    "Company",
    "CustomerType",
    "ReservationStatus",
]

# Categorical features selected for the initial ADR regression model.
# Assumption: prediction happens when the booking is initially created.
REGRESSION_CATEGORICAL_FEATURES = [
    "ArrivalDateMonth",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "ReservedRoomType",
    "DepositType",
    "Agent",
    "Company",
    "CustomerType",
]

# Categorical columns excluded from the initial regression features.
REGRESSION_EXCLUDED_CATEGORICAL = [
    "AssignedRoomType",
    "ReservationStatus",
]

DATE_COLUMNS = [
    "ReservationStatusDate",
]

# Regression target.
REGRESSION_TARGET = "ADR"

BOOKING_TIME_NUMERIC = [
    "LeadTime",
    "ArrivalDateYear",
    "ArrivalDateWeekNumber",
    "ArrivalDateDayOfMonth",
    "StaysInWeekendNights",
    "StaysInWeekNights",
    "Adults",
    "Children",
    "Babies",
    "IsRepeatedGuest",
    "PreviousCancellations",
    "PreviousBookingsNotCanceled",
]

BOOKING_TIME_CATEGORICAL = [
    "ArrivalDateMonth",
    "Meal",
    "Country",
    "MarketSegment",
    "DistributionChannel",
    "ReservedRoomType",
    "DepositType",
    "Agent",
    "Company",
    "CustomerType",
]

BORDERLINE_NUMERIC = [
    "TotalOfSpecialRequests",
    "RequiredCarParkingSpaces",
]

LEAKY_COLUMNS = [
    "IsCanceled",
    "ReservationStatus",
    "ReservationStatusDate",
    "AssignedRoomType",
    "BookingChanges",
    "DaysInWaitingList",
]




   
